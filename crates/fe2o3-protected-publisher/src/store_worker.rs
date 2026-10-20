use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Instant as StdInstant;

use tokio::sync::{oneshot, watch};
use tokio::time::Instant;

use crate::PublisherError;
use crate::oidc::PublisherRequest;
use crate::receipt::ReceiptSigner;
use crate::store::{DurableStore, IssueInput};

pub(crate) const STORE_WORKER_COUNT: usize = 1;

pub(crate) struct StoreIssue {
    pub(crate) request_key_sha256: String,
    pub(crate) stable_authorization_sha256: String,
    pub(crate) request_identity: String,
    pub(crate) request_sha256: String,
    pub(crate) request_body: Vec<u8>,
    pub(crate) request: PublisherRequest,
    pub(crate) issued_at: i64,
    pub(crate) signature_domain: String,
    pub(crate) signer: Arc<dyn ReceiptSigner>,
}

impl StoreIssue {
    fn execute(
        &self,
        store: &mut DurableStore,
        deadline: StdInstant,
    ) -> Result<Vec<u8>, PublisherError> {
        store.issue_until(
            IssueInput {
                request_key_sha256: &self.request_key_sha256,
                stable_authorization_sha256: &self.stable_authorization_sha256,
                request_identity: &self.request_identity,
                request_sha256: &self.request_sha256,
                request_body: &self.request_body,
                request: &self.request,
                issued_at: self.issued_at,
                signature_domain: &self.signature_domain,
                signer: self.signer.as_ref(),
            },
            deadline,
        )
    }
}

enum Command {
    Issue {
        input: Box<StoreIssue>,
        deadline: StdInstant,
        response: oneshot::Sender<Result<Vec<u8>, PublisherError>>,
    },
    #[cfg(test)]
    Count(oneshot::Sender<usize>),
    #[cfg(test)]
    Break(oneshot::Sender<()>),
}

pub(crate) struct StoreWorker {
    sender: Mutex<Option<SyncSender<Command>>>,
    stopping: Arc<AtomicBool>,
    stopped: watch::Receiver<bool>,
}

impl StoreWorker {
    pub(crate) fn spawn(
        mut store: DurableStore,
        queue_capacity: usize,
    ) -> Result<Self, PublisherError> {
        if STORE_WORKER_COUNT != 1 || queue_capacity == 0 {
            return Err(PublisherError::Config);
        }
        let (sender, receiver) = mpsc::sync_channel(queue_capacity);
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = stopping.clone();
        let (stopped_tx, stopped) = watch::channel(false);
        std::thread::Builder::new()
            .name("fe2o3-publisher-store-0".into())
            .spawn(move || {
                while let Ok(command) = receiver.recv() {
                    if worker_stopping.load(Ordering::Acquire) {
                        reject(command);
                        continue;
                    }
                    match command {
                        Command::Issue {
                            input,
                            deadline,
                            response,
                        } => {
                            let _ = response.send(input.execute(&mut store, deadline));
                        }
                        #[cfg(test)]
                        Command::Count(response) => {
                            let _ = response.send(store.count());
                        }
                        #[cfg(test)]
                        Command::Break(response) => {
                            store.break_for_test();
                            let _ = response.send(());
                        }
                    }
                }
                // Normal completion retires the ledger, not just its command loop.
                drop(store);
                let _ = stopped_tx.send(true);
            })
            .map_err(|_| PublisherError::Store)?;
        Ok(Self {
            sender: Mutex::new(Some(sender)),
            stopping,
            stopped,
        })
    }

    pub(crate) async fn issue_until(
        &self,
        input: StoreIssue,
        deadline: Instant,
    ) -> Result<Vec<u8>, PublisherError> {
        if Instant::now() >= deadline || self.stopping.load(Ordering::Acquire) {
            return Err(PublisherError::Store);
        }
        let (response, result) = oneshot::channel();
        self.try_send(Command::Issue {
            input: Box::new(input),
            deadline: deadline.into_std(),
            response,
        })?;
        tokio::time::timeout_at(deadline, result)
            .await
            .map_err(|_| PublisherError::Store)?
            .map_err(|_| PublisherError::Store)?
    }

    pub(crate) async fn shutdown_until(&self, deadline: Instant) -> bool {
        self.begin_shutdown();
        let mut stopped = self.stopped.clone();
        if *stopped.borrow() {
            return true;
        }
        tokio::time::timeout_at(deadline, async move {
            while !*stopped.borrow() {
                if stopped.changed().await.is_err() {
                    break;
                }
            }
        })
        .await
        .is_ok()
            && *self.stopped.borrow()
    }

    fn begin_shutdown(&self) {
        self.stopping.store(true, Ordering::Release);
        if let Ok(mut sender) = self.sender.lock() {
            sender.take();
        }
    }

    fn try_send(&self, command: Command) -> Result<(), PublisherError> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| PublisherError::Store)?
            .as_ref()
            .cloned()
            .ok_or(PublisherError::Store)?;
        sender.try_send(command).map_err(|error| match error {
            TrySendError::Full(_) | TrySendError::Disconnected(_) => PublisherError::Store,
        })
    }

    #[cfg(test)]
    pub(crate) async fn count(&self) -> usize {
        let (response, result) = oneshot::channel();
        self.try_send(Command::Count(response)).unwrap();
        result.await.unwrap()
    }

    #[cfg(test)]
    pub(crate) async fn break_for_test(&self) {
        let (response, result) = oneshot::channel();
        self.try_send(Command::Break(response)).unwrap();
        result.await.unwrap();
    }
}

impl Drop for StoreWorker {
    fn drop(&mut self) {
        self.begin_shutdown();
    }
}

fn reject(command: Command) {
    match command {
        Command::Issue { response, .. } => {
            let _ = response.send(Err(PublisherError::Store));
        }
        #[cfg(test)]
        Command::Count(response) => {
            drop(response);
        }
        #[cfg(test)]
        Command::Break(response) => {
            drop(response);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::test_support::secure_tempdir;

    #[tokio::test]
    async fn normal_shutdown_signals_only_after_store_retirement() {
        let temp = secure_tempdir();
        let path = temp.path().join("retirement.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        store.set_before_unlock(move || {
            let _ = entered_tx.send(());
            let _ = release_rx.recv();
        });
        let worker = Arc::new(StoreWorker::spawn(store, 1).unwrap());
        let stopping = worker.clone();
        let shutdown = tokio::spawn(async move {
            stopping
                .shutdown_until(Instant::now() + Duration::from_secs(5))
                .await
        });

        tokio::time::timeout(Duration::from_secs(5), entered_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(!*worker.stopped.borrow());
        assert!(DurableStore::open(&path).is_err());
        release_tx.send(()).unwrap();
        assert!(shutdown.await.unwrap());
        assert!(*worker.stopped.borrow());
        let reopened = DurableStore::open(&path).unwrap();
        assert_eq!(reopened.count(), 0);
    }
}
