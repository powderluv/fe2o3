use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

#[cfg(test)]
use std::time::Duration;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::PublisherError;
use crate::ServiceConfig;
use crate::bounds::{
    MAX_LEDGER_BYTES, MAX_LEDGER_FRAME_BYTES, MAX_LEDGER_RECORD_BYTES,
    MAX_LEDGER_REQUEST_BASE64_BYTES, MAX_LEDGER_REQUEST_BODY_BYTES,
    MAX_LEDGER_RESPONSE_BASE64_BYTES, MAX_LEDGER_RESPONSE_BODY_BYTES, MAX_RECEIPT_BASE64_BYTES,
    MAX_RECEIPT_BYTES, MAX_STORE_RECEIPTS, MIN_LEDGER_BYTES,
};
use crate::canonical::{canonical_bytes, parse_canonical, parse_canonical_with_string_limit};
use crate::oidc::PublisherRequest;
use crate::receipt::{
    ReceiptArtifact, ReceiptSigner, build_artifact, raw_request_sha256, request_identity,
};
use crate::secure_fs::{FileIdentity, SecureLocation};

const LEDGER_MAGIC: &[u8] = b"fe2o3-protected-publisher-ledger-v3\0";
#[cfg(test)]
const LEGACY_LEDGER_MAGICS: [&[u8]; 2] = [
    b"fe2o3-protected-publisher-ledger-v1\0",
    b"fe2o3-protected-publisher-ledger-v2\0",
];
const FRAME_MAGIC: &[u8; 8] = b"F2O3REC3";
const FRAME_VERSION: u32 = 3;
const FRAME_PREFIX_BYTES: usize = 8 + 4 + 4 + 8 + 32;
const FRAME_HASH_BYTES: usize = 32;
const FRAME_TRAILER_MAGIC: &[u8; 8] = b"F2O3CMT3";
const FRAME_TRAILER_BYTES: usize = 8 + 8 + 8 + 32 + 32 + 8;
const MIN_FRAME_BYTES: usize = FRAME_PREFIX_BYTES + 1 + FRAME_HASH_BYTES + FRAME_TRAILER_BYTES;
const CHECKPOINT_MAGIC: &[u8; 8] = b"F2O3CP3!";
const CHECKPOINT_END_MAGIC: &[u8; 8] = b"F2O3END3";
const CHECKPOINT_VERSION: u32 = 3;
const CHECKPOINT_COPY_BYTES: usize = 8 + 4 + 4 + 8 + 8 + 32 + 32 + 8;
const CHECKPOINT_COPIES: usize = 3;
const CHECKPOINT_REGION_BYTES: usize = CHECKPOINT_COPY_BYTES * CHECKPOINT_COPIES;
const ZERO_HASH: [u8; 32] = [0; 32];
const MAX_INDEXED_READ_INTERRUPTS: usize = 8;

#[cfg(test)]
type BeforeAdmissionHook = Box<dyn FnOnce(&mut Vec<u8>) + Send>;

#[cfg(test)]
type BeforeIndexedDecodeHook = Box<dyn FnOnce(&mut Vec<u8>) + Send>;

#[cfg(test)]
enum IndexedReadFault {
    ErrorOnCall { call: usize, calls: usize },
    ErrorAfterBytes { bytes: usize, observed: usize },
    EofOnCall { call: usize, calls: usize },
    Interrupt { remaining: usize },
}

pub struct DurableStore {
    file: File,
    policy: StorePolicy,
    location: SecureLocation,
    identity: FileIdentity,
    checkpoint_offset: u64,
    checkpoint_generation: u64,
    header_len: u64,
    tail_offset: u64,
    next_sequence: u64,
    tail_hash: [u8; 32],
    by_request_key: HashMap<String, EntryIndex>,
    request_identities: HashSet<String>,
    request_digests: HashSet<String>,
    evidence_identities: HashSet<String>,
    poisoned: bool,
    #[cfg(test)]
    commit_delay: Duration,
    #[cfg(test)]
    maximum_write_chunk: usize,
    #[cfg(test)]
    fail_write_after: Option<(usize, i32)>,
    #[cfg(test)]
    fail_checkpoint_write_after: Option<(usize, i32)>,
    #[cfg(test)]
    after_sync: Option<Box<dyn FnOnce() + Send>>,
    #[cfg(test)]
    before_admission: Option<BeforeAdmissionHook>,
    #[cfg(test)]
    after_admission: Option<Box<dyn FnOnce() + Send>>,
    #[cfg(test)]
    maximum_indexed_read_chunk: usize,
    #[cfg(test)]
    indexed_read_fault: Option<IndexedReadFault>,
    #[cfg(test)]
    before_indexed_decode: Option<BeforeIndexedDecodeHook>,
    #[cfg(test)]
    before_unlock: Option<Box<dyn FnOnce() + Send>>,
}

#[derive(Clone)]
struct EntryIndex {
    frame_offset: u64,
    frame_length: u64,
    sequence: u64,
    previous_hash: [u8; 32],
    frame_hash: [u8; 32],
    request_key_sha256: String,
    request_identity: String,
    request_sha256: String,
    stable_authorization_sha256: String,
}

struct IndexedReceipt {
    record: LedgerRecord,
    request_body: Vec<u8>,
    response_body: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct StorePolicy {
    max_receipts: u64,
    max_ledger_bytes: u64,
    ledger_domain: String,
}

impl StorePolicy {
    pub(crate) fn from_config(config: &ServiceConfig) -> Result<Self, PublisherError> {
        let policy = Self {
            max_receipts: config.max_receipts,
            max_ledger_bytes: config.max_ledger_bytes,
            ledger_domain: config.service_identity()?,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<(), PublisherError> {
        if self.max_receipts == 0
            || self.max_receipts > MAX_STORE_RECEIPTS
            || !(MIN_LEDGER_BYTES..=MAX_LEDGER_BYTES).contains(&self.max_ledger_bytes)
            || self.ledger_domain.len() != 64
            || !self.ledger_domain.is_ascii()
        {
            return Err(PublisherError::Config);
        }
        Ok(())
    }

    #[cfg(test)]
    fn test_default() -> Self {
        Self {
            max_receipts: 4096,
            max_ledger_bytes: 64 * 1024 * 1024,
            ledger_domain: format!("t{}", "1".repeat(63)),
        }
    }
}

pub(crate) struct IssueInput<'a> {
    pub request_key_sha256: &'a str,
    pub stable_authorization_sha256: &'a str,
    pub request_identity: &'a str,
    pub request_sha256: &'a str,
    pub request_body: &'a [u8],
    pub request: &'a PublisherRequest,
    pub issued_at: i64,
    pub signature_domain: &'a str,
    pub signer: &'a dyn ReceiptSigner,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
struct LedgerRecord {
    evidence_identity: String,
    issued_at: i64,
    record_domain: String,
    request_body_base64: String,
    request_identity: String,
    request_key_sha256: String,
    request_sha256: String,
    response_body_base64: String,
    schema_version: u32,
    stable_authorization_sha256: String,
}

struct DecodedFrame {
    frame_hash: [u8; 32],
    frame_length: u64,
    record: LedgerRecord,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CommitCheckpoint {
    generation: u64,
    tail_offset: u64,
    tail_hash: [u8; 32],
}

impl DurableStore {
    #[cfg(test)]
    pub fn open(path: &Path) -> Result<Self, PublisherError> {
        Self::open_with_policy(path, StorePolicy::test_default())
    }

    pub(crate) fn open_with_policy(
        path: &Path,
        policy: StorePolicy,
    ) -> Result<Self, PublisherError> {
        policy.validate()?;
        let location = SecureLocation::open(path)?;
        let header = ledger_header(&policy)?;
        let checkpoint_offset = header
            .len()
            .checked_sub(CHECKPOINT_REGION_BYTES)
            .ok_or(PublisherError::Store)? as u64;
        let (file, identity) = location.open_or_create_ledger(&header)?;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(PublisherError::Store);
        }

        let mut store = Self {
            file,
            policy,
            location,
            identity,
            checkpoint_offset,
            checkpoint_generation: 0,
            header_len: header.len() as u64,
            tail_offset: header.len() as u64,
            next_sequence: 1,
            tail_hash: ZERO_HASH,
            by_request_key: HashMap::new(),
            request_identities: HashSet::new(),
            request_digests: HashSet::new(),
            evidence_identities: HashSet::new(),
            poisoned: false,
            #[cfg(test)]
            commit_delay: Duration::ZERO,
            #[cfg(test)]
            maximum_write_chunk: usize::MAX,
            #[cfg(test)]
            fail_write_after: None,
            #[cfg(test)]
            fail_checkpoint_write_after: None,
            #[cfg(test)]
            after_sync: None,
            #[cfg(test)]
            before_admission: None,
            #[cfg(test)]
            after_admission: None,
            #[cfg(test)]
            maximum_indexed_read_chunk: usize::MAX,
            #[cfg(test)]
            indexed_read_fault: None,
            #[cfg(test)]
            before_indexed_decode: None,
            #[cfg(test)]
            before_unlock: None,
        };
        store.verify_identity()?;
        store.replay(&header)?;
        store.verify_identity()?;
        Ok(store)
    }

    fn replay(&mut self, expected_header: &[u8]) -> Result<(), PublisherError> {
        let length = self
            .file
            .metadata()
            .map_err(|_| PublisherError::Store)?
            .len();
        if length > self.policy.max_ledger_bytes || length < self.header_len {
            return Err(PublisherError::Store);
        }
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|_| PublisherError::Store)?;
        let mut header = vec![0; expected_header.len()];
        self.file
            .read_exact(&mut header)
            .map_err(|_| PublisherError::Store)?;
        let checkpoint_offset =
            usize::try_from(self.checkpoint_offset).map_err(|_| PublisherError::Store)?;
        if header[..checkpoint_offset] != expected_header[..checkpoint_offset] {
            return Err(PublisherError::Store);
        }

        let checkpoint = select_checkpoint(
            &header[checkpoint_offset..],
            self.header_len,
            self.policy.max_receipts,
            self.policy.max_ledger_bytes,
        )?;
        if checkpoint.tail_offset > length {
            return Err(PublisherError::Store);
        }

        let mut offset = self.header_len;
        while offset < checkpoint.tail_offset {
            let remaining = checkpoint.tail_offset - offset;
            if remaining < FRAME_PREFIX_BYTES as u64 {
                return Err(PublisherError::Store);
            }
            let decoded = self.read_frame(offset, remaining)?;
            self.index_record(
                offset,
                decoded.frame_length,
                self.next_sequence,
                self.tail_hash,
                decoded.frame_hash,
                &decoded.record,
            )?;
            self.tail_hash = decoded.frame_hash;
            self.next_sequence = self
                .next_sequence
                .checked_add(1)
                .ok_or(PublisherError::Store)?;
            offset = offset
                .checked_add(decoded.frame_length)
                .ok_or(PublisherError::Store)?;
            self.tail_offset = offset;
        }
        if offset != checkpoint.tail_offset
            || self.next_sequence.checked_sub(1) != Some(checkpoint.generation)
            || self.tail_hash != checkpoint.tail_hash
        {
            return Err(PublisherError::Store);
        }
        self.checkpoint_generation = checkpoint.generation;
        if length > checkpoint.tail_offset {
            self.recover_torn_tail(checkpoint.tail_offset)?;
        }
        Ok(())
    }

    fn recover_torn_tail(&mut self, offset: u64) -> Result<(), PublisherError> {
        self.file
            .set_len(offset)
            .map_err(|_| PublisherError::Store)?;
        self.file.sync_data().map_err(|_| PublisherError::Store)?;
        self.tail_offset = offset;
        Ok(())
    }

    fn read_frame(&mut self, offset: u64, remaining: u64) -> Result<DecodedFrame, PublisherError> {
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|_| PublisherError::Store)?;
        let mut prefix = [0u8; FRAME_PREFIX_BYTES];
        self.file
            .read_exact(&mut prefix)
            .map_err(|_| PublisherError::Store)?;
        let frame_length = frame_length_from_prefix(&prefix, self.next_sequence, self.tail_hash)?;
        if remaining < frame_length as u64 {
            return Err(PublisherError::Store);
        }
        let mut frame = Vec::with_capacity(frame_length);
        frame.extend_from_slice(&prefix);
        let rest_length = frame_length
            .checked_sub(FRAME_PREFIX_BYTES)
            .ok_or(PublisherError::Store)?;
        let mut rest = vec![0; rest_length];
        self.file
            .read_exact(&mut rest)
            .map_err(|_| PublisherError::Store)?;
        frame.extend_from_slice(&rest);
        decode_frame(&frame, self.next_sequence, self.tail_hash)
    }

    fn index_record(
        &mut self,
        frame_offset: u64,
        frame_length: u64,
        sequence: u64,
        previous_hash: [u8; 32],
        frame_hash: [u8; 32],
        record: &LedgerRecord,
    ) -> Result<(), PublisherError> {
        if self.by_request_key.len() as u64 >= self.policy.max_receipts
            || self.by_request_key.contains_key(&record.request_key_sha256)
            || !self
                .request_identities
                .insert(record.request_identity.clone())
            || !self.request_digests.insert(record.request_sha256.clone())
            || !self
                .evidence_identities
                .insert(record.evidence_identity.clone())
        {
            return Err(PublisherError::Store);
        }
        self.by_request_key.insert(
            record.request_key_sha256.clone(),
            EntryIndex {
                frame_offset,
                frame_length,
                sequence,
                previous_hash,
                frame_hash,
                request_key_sha256: record.request_key_sha256.clone(),
                request_identity: record.request_identity.clone(),
                request_sha256: record.request_sha256.clone(),
                stable_authorization_sha256: record.stable_authorization_sha256.clone(),
            },
        );
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn issue(&mut self, input: IssueInput<'_>) -> Result<Vec<u8>, PublisherError> {
        self.issue_until(input, Instant::now() + Duration::from_secs(60))
    }

    pub(crate) fn issue_until(
        &mut self,
        input: IssueInput<'_>,
        deadline: Instant,
    ) -> Result<Vec<u8>, PublisherError> {
        if self.poisoned {
            return Err(PublisherError::Store);
        }
        check_deadline(deadline)?;
        self.verify_identity()?;
        validate_issue_input(&input)?;
        check_deadline(deadline)?;

        if let Some(existing) = self.by_request_key.get(input.request_key_sha256).cloned() {
            let indexed = self.load_indexed(input.request_key_sha256, &existing)?;
            if indexed.record.request_key_sha256 != input.request_key_sha256
                || indexed.record.request_identity != input.request_identity
                || indexed.record.request_sha256 != input.request_sha256
                || indexed.record.stable_authorization_sha256 != input.stable_authorization_sha256
                || indexed.request_body != input.request_body
            {
                return Err(PublisherError::ReplayConflict);
            }
            check_deadline(deadline)?;
            return Ok(indexed.response_body);
        }
        if self.request_identities.contains(input.request_identity)
            || self.request_digests.contains(input.request_sha256)
        {
            return Err(PublisherError::ReplayConflict);
        }
        if self.by_request_key.len() as u64 >= self.policy.max_receipts {
            return Err(PublisherError::Store);
        }
        check_deadline(deadline)?;

        let ReceiptArtifact {
            evidence_identity,
            response,
        } = build_artifact(
            input.request,
            input.request_identity,
            input.request_sha256,
            input.issued_at,
            input.signature_domain,
            input.signer,
        )?;
        let record = LedgerRecord {
            evidence_identity,
            issued_at: input.issued_at,
            record_domain: "fe2o3-protected-publisher-ledger-record-v1".into(),
            request_body_base64: base64::engine::general_purpose::STANDARD
                .encode(input.request_body),
            request_identity: input.request_identity.into(),
            request_key_sha256: input.request_key_sha256.into(),
            request_sha256: input.request_sha256.into(),
            response_body_base64: base64::engine::general_purpose::STANDARD.encode(&response),
            schema_version: 1,
            stable_authorization_sha256: input.stable_authorization_sha256.into(),
        };
        let payload =
            canonical_bytes(&serde_json::to_value(&record).map_err(|_| PublisherError::Store)?)
                .map_err(|_| PublisherError::Store)?;
        let frame = encode_frame(self.next_sequence, self.tail_hash, &payload)?;
        #[cfg(test)]
        let mut frame = frame;
        #[cfg(test)]
        if let Some(hook) = self.before_admission.take() {
            hook(&mut frame);
        }
        // Append admission uses the exact decoder and record validation used by restart.
        let decoded = decode_frame(&frame, self.next_sequence, self.tail_hash)?;
        if decoded.record != record {
            return Err(PublisherError::Store);
        }
        if self
            .evidence_identities
            .contains(&decoded.record.evidence_identity)
        {
            return Err(PublisherError::ReplayConflict);
        }
        let new_tail = self
            .tail_offset
            .checked_add(decoded.frame_length)
            .filter(|tail| *tail <= self.policy.max_ledger_bytes)
            .ok_or(PublisherError::Store)?;
        check_deadline(deadline)?;
        self.verify_identity()?;

        // Admission ends here. Synchronous writes and fdatasync are not cancellable.
        #[cfg(test)]
        if let Some(hook) = self.after_admission.take() {
            hook();
        }
        #[cfg(test)]
        std::thread::sleep(self.commit_delay);
        if self.append_authoritatively(&frame).is_err() {
            self.poisoned = true;
            return Err(PublisherError::Store);
        }
        let frame_offset = self.tail_offset;
        let frame_sequence = self.next_sequence;
        let previous_hash = self.tail_hash;
        let checkpoint = CommitCheckpoint {
            generation: frame_sequence,
            tail_offset: new_tail,
            tail_hash: decoded.frame_hash,
        };
        if self.persist_checkpoint(checkpoint).is_err() {
            self.poisoned = true;
            return Err(PublisherError::Store);
        }
        self.checkpoint_generation = checkpoint.generation;
        self.tail_offset = new_tail;
        self.tail_hash = decoded.frame_hash;
        self.next_sequence = frame_sequence.checked_add(1).ok_or(PublisherError::Store)?;
        if self
            .index_record(
                frame_offset,
                decoded.frame_length,
                frame_sequence,
                previous_hash,
                decoded.frame_hash,
                &decoded.record,
            )
            .is_err()
        {
            self.poisoned = true;
            return Err(PublisherError::Store);
        }
        Ok(response)
    }

    fn append_authoritatively(&mut self, frame: &[u8]) -> Result<(), PublisherError> {
        self.file
            .seek(SeekFrom::Start(self.tail_offset))
            .map_err(|_| PublisherError::Store)?;
        let mut written = 0usize;
        while written < frame.len() {
            #[cfg(test)]
            if self
                .fail_write_after
                .is_some_and(|(threshold, _)| written >= threshold)
            {
                let (_, errno) = self.fail_write_after.unwrap();
                let _injected = std::io::Error::from_raw_os_error(errno);
                return Err(PublisherError::Store);
            }
            #[cfg(test)]
            let end = frame
                .len()
                .min(written.saturating_add(self.maximum_write_chunk));
            #[cfg(not(test))]
            let end = frame.len();
            let count = self
                .file
                .write(&frame[written..end])
                .map_err(|_| PublisherError::Store)?;
            if count == 0 {
                return Err(PublisherError::Store);
            }
            written = written.checked_add(count).ok_or(PublisherError::Store)?;
        }
        self.file.sync_data().map_err(|_| PublisherError::Store)?;
        #[cfg(test)]
        if let Some(hook) = self.after_sync.take() {
            hook();
        }
        self.verify_identity()
    }

    fn persist_checkpoint(&mut self, checkpoint: CommitCheckpoint) -> Result<(), PublisherError> {
        if checkpoint.generation
            != self
                .checkpoint_generation
                .checked_add(1)
                .ok_or(PublisherError::Store)?
            || checkpoint.tail_offset <= self.tail_offset
            || checkpoint.tail_offset > self.policy.max_ledger_bytes
            || checkpoint.tail_hash == ZERO_HASH
        {
            return Err(PublisherError::Store);
        }
        let encoded = encode_checkpoint(checkpoint);
        #[cfg(test)]
        let mut total_written = 0usize;
        for copy in 0..CHECKPOINT_COPIES {
            let copy_offset = self
                .checkpoint_offset
                .checked_add(
                    u64::try_from(
                        copy.checked_mul(CHECKPOINT_COPY_BYTES)
                            .ok_or(PublisherError::Store)?,
                    )
                    .map_err(|_| PublisherError::Store)?,
                )
                .ok_or(PublisherError::Store)?;
            let mut written = 0usize;
            while written < encoded.len() {
                #[cfg(test)]
                if self
                    .fail_checkpoint_write_after
                    .is_some_and(|(threshold, _)| total_written >= threshold)
                {
                    let (_, errno) = self.fail_checkpoint_write_after.unwrap();
                    let _injected = std::io::Error::from_raw_os_error(errno);
                    return Err(PublisherError::Store);
                }
                let offset = copy_offset
                    .checked_add(u64::try_from(written).map_err(|_| PublisherError::Store)?)
                    .ok_or(PublisherError::Store)?;
                #[cfg(test)]
                let end = encoded
                    .len()
                    .min(written.saturating_add(self.maximum_write_chunk));
                #[cfg(not(test))]
                let end = encoded.len();
                let count = loop {
                    match self.file.write_at(&encoded[written..end], offset) {
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                        result => break result.map_err(|_| PublisherError::Store)?,
                    }
                };
                if count == 0 {
                    return Err(PublisherError::Store);
                }
                written = written.checked_add(count).ok_or(PublisherError::Store)?;
                #[cfg(test)]
                {
                    total_written = total_written
                        .checked_add(count)
                        .ok_or(PublisherError::Store)?;
                }
            }
        }
        self.file.sync_data().map_err(|_| PublisherError::Store)?;
        self.verify_identity()
    }

    fn load_indexed(
        &mut self,
        expected_request_key_sha256: &str,
        index: &EntryIndex,
    ) -> Result<IndexedReceipt, PublisherError> {
        if self.poisoned {
            return Err(PublisherError::Store);
        }
        let result = self.load_indexed_observational(expected_request_key_sha256, index);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn load_indexed_observational(
        &mut self,
        expected_request_key_sha256: &str,
        index: &EntryIndex,
    ) -> Result<IndexedReceipt, PublisherError> {
        if expected_request_key_sha256 != index.request_key_sha256 {
            return Err(PublisherError::Store);
        }
        self.verify_identity()?;
        let length = self
            .file
            .metadata()
            .map_err(|_| PublisherError::Store)?
            .len();
        let frame_end = index
            .frame_offset
            .checked_add(index.frame_length)
            .ok_or(PublisherError::Store)?;
        if index.frame_offset < self.header_len
            || index.frame_length < MIN_FRAME_BYTES as u64
            || index.frame_length > MAX_LEDGER_FRAME_BYTES as u64
            || frame_end > self.tail_offset
            || frame_end > length
        {
            return Err(PublisherError::Store);
        }

        let mut prefix = [0u8; FRAME_PREFIX_BYTES];
        self.read_exact_indexed_at(&mut prefix, index.frame_offset)?;
        let frame_length = frame_length_from_prefix(&prefix, index.sequence, index.previous_hash)?;
        if frame_length as u64 != index.frame_length {
            return Err(PublisherError::Store);
        }
        let mut frame = Vec::with_capacity(frame_length);
        frame.extend_from_slice(&prefix);
        let mut rest = vec![0u8; frame_length - FRAME_PREFIX_BYTES];
        self.read_exact_indexed_at(
            &mut rest,
            index
                .frame_offset
                .checked_add(FRAME_PREFIX_BYTES as u64)
                .ok_or(PublisherError::Store)?,
        )?;
        frame.extend_from_slice(&rest);
        #[cfg(test)]
        if let Some(hook) = self.before_indexed_decode.take() {
            hook(&mut frame);
        }

        let decoded = decode_frame(&frame, index.sequence, index.previous_hash)?;
        if decoded.frame_length != index.frame_length
            || decoded.frame_hash != index.frame_hash
            || decoded.record.request_key_sha256 != expected_request_key_sha256
            || decoded.record.request_key_sha256 != index.request_key_sha256
            || decoded.record.request_identity != index.request_identity
            || decoded.record.request_sha256 != index.request_sha256
            || decoded.record.stable_authorization_sha256 != index.stable_authorization_sha256
        {
            return Err(PublisherError::Store);
        }
        let request_body = decode_base64(
            &decoded.record.request_body_base64,
            MAX_LEDGER_REQUEST_BASE64_BYTES,
            MAX_LEDGER_REQUEST_BODY_BYTES,
        )?;
        let response_body = decode_base64(
            &decoded.record.response_body_base64,
            MAX_LEDGER_RESPONSE_BASE64_BYTES,
            MAX_LEDGER_RESPONSE_BODY_BYTES,
        )?;
        self.verify_identity()?;
        Ok(IndexedReceipt {
            record: decoded.record,
            request_body,
            response_body,
        })
    }

    fn read_exact_indexed_at(
        &mut self,
        buffer: &mut [u8],
        offset: u64,
    ) -> Result<(), PublisherError> {
        let mut filled = 0usize;
        let mut interruptions = 0usize;
        while filled < buffer.len() {
            let read_offset = offset
                .checked_add(filled as u64)
                .ok_or(PublisherError::Store)?;
            match self.read_indexed_at(&mut buffer[filled..], read_offset) {
                Ok(0) => return Err(PublisherError::Store),
                Ok(count) if count <= buffer.len() - filled => {
                    filled = filled.checked_add(count).ok_or(PublisherError::Store)?;
                }
                Ok(_) => return Err(PublisherError::Store),
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {
                    interruptions = interruptions.checked_add(1).ok_or(PublisherError::Store)?;
                    if interruptions > MAX_INDEXED_READ_INTERRUPTS {
                        return Err(PublisherError::Store);
                    }
                }
                Err(_) => return Err(PublisherError::Store),
            }
        }
        Ok(())
    }

    fn read_indexed_at(&mut self, buffer: &mut [u8], offset: u64) -> std::io::Result<usize> {
        #[cfg(test)]
        let mut limit = buffer.len().min(self.maximum_indexed_read_chunk);
        #[cfg(not(test))]
        let limit = buffer.len();

        #[cfg(test)]
        if let Some(fault) = self.indexed_read_fault.as_mut() {
            match fault {
                IndexedReadFault::ErrorOnCall { call, calls } => {
                    *calls += 1;
                    if *calls == *call {
                        return Err(std::io::Error::from_raw_os_error(libc::EIO));
                    }
                }
                IndexedReadFault::ErrorAfterBytes { bytes, observed } => {
                    if *observed >= *bytes {
                        return Err(std::io::Error::from_raw_os_error(libc::EIO));
                    }
                    limit = limit.min(*bytes - *observed);
                }
                IndexedReadFault::EofOnCall { call, calls } => {
                    *calls += 1;
                    if *calls == *call {
                        return Ok(0);
                    }
                }
                IndexedReadFault::Interrupt { remaining } => {
                    if *remaining > 0 {
                        *remaining -= 1;
                        return Err(std::io::Error::from(std::io::ErrorKind::Interrupted));
                    }
                }
            }
        }

        let count = self.file.read_at(&mut buffer[..limit], offset)?;
        #[cfg(test)]
        if let Some(IndexedReadFault::ErrorAfterBytes { observed, .. }) =
            self.indexed_read_fault.as_mut()
        {
            *observed = observed.saturating_add(count);
        }
        Ok(count)
    }

    fn verify_identity(&self) -> Result<(), PublisherError> {
        self.location
            .verify_ledger_entry(self.identity)
            .map_err(|_| PublisherError::Store)?;
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        if unsafe { libc::fstat(self.file.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
            return Err(PublisherError::Store);
        }
        let stat = unsafe { stat.assume_init() };
        if stat.st_dev != self.identity.dev
            || stat.st_ino != self.identity.ino
            || stat.st_mode != self.identity.mode
            || stat.st_uid != self.identity.uid
            || stat.st_gid != self.identity.gid
            || stat.st_nlink != 1
        {
            return Err(PublisherError::Store);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.by_request_key.len()
    }

    #[cfg(test)]
    pub(crate) fn break_for_test(&mut self) {
        self.poisoned = true;
    }

    #[cfg(test)]
    pub(crate) fn set_commit_delay(&mut self, delay: Duration) {
        self.commit_delay = delay;
    }

    #[cfg(test)]
    fn set_maximum_write_chunk(&mut self, bytes: usize) {
        self.maximum_write_chunk = bytes.max(1);
    }

    #[cfg(test)]
    fn fail_write_after(&mut self, bytes: usize, errno: i32) {
        self.fail_write_after = Some((bytes, errno));
    }

    #[cfg(test)]
    fn fail_checkpoint_write_after(&mut self, bytes: usize, errno: i32) {
        self.fail_checkpoint_write_after = Some((bytes, errno));
    }

    #[cfg(test)]
    fn set_after_sync(&mut self, hook: impl FnOnce() + Send + 'static) {
        self.after_sync = Some(Box::new(hook));
    }

    #[cfg(test)]
    fn set_before_admission(&mut self, hook: impl FnOnce(&mut Vec<u8>) + Send + 'static) {
        self.before_admission = Some(Box::new(hook));
    }

    #[cfg(test)]
    pub(crate) fn set_after_admission(&mut self, hook: impl FnOnce() + Send + 'static) {
        self.after_admission = Some(Box::new(hook));
    }

    #[cfg(test)]
    fn set_maximum_indexed_read_chunk(&mut self, bytes: usize) {
        self.maximum_indexed_read_chunk = bytes.max(1);
    }

    #[cfg(test)]
    fn fail_indexed_read_on_call(&mut self, call: usize) {
        self.indexed_read_fault = Some(IndexedReadFault::ErrorOnCall { call, calls: 0 });
    }

    #[cfg(test)]
    fn fail_indexed_read_after(&mut self, bytes: usize) {
        self.indexed_read_fault = Some(IndexedReadFault::ErrorAfterBytes { bytes, observed: 0 });
    }

    #[cfg(test)]
    fn eof_indexed_read_on_call(&mut self, call: usize) {
        self.indexed_read_fault = Some(IndexedReadFault::EofOnCall { call, calls: 0 });
    }

    #[cfg(test)]
    fn interrupt_indexed_reads(&mut self, count: usize) {
        self.indexed_read_fault = Some(IndexedReadFault::Interrupt { remaining: count });
    }

    #[cfg(test)]
    fn set_before_indexed_decode(&mut self, hook: impl FnOnce(&mut Vec<u8>) + Send + 'static) {
        self.before_indexed_decode = Some(Box::new(hook));
    }

    #[cfg(test)]
    pub(crate) fn set_before_unlock(&mut self, hook: impl FnOnce() + Send + 'static) {
        self.before_unlock = Some(Box::new(hook));
    }
}

impl Drop for DurableStore {
    fn drop(&mut self) {
        #[cfg(test)]
        if let Some(hook) = self.before_unlock.take() {
            hook();
        }
        // Unlock the shared open-file description before close, including any
        // transient fork inheritance that has not reached close-on-exec yet.
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

fn ledger_header(policy: &StorePolicy) -> Result<Vec<u8>, PublisherError> {
    let static_bytes = LEDGER_MAGIC
        .len()
        .checked_add(policy.ledger_domain.len())
        .and_then(|length| length.checked_add(1))
        .ok_or(PublisherError::Store)?;
    let header_len = static_bytes
        .checked_add(CHECKPOINT_REGION_BYTES)
        .ok_or(PublisherError::Store)?;
    let tail_offset = u64::try_from(header_len).map_err(|_| PublisherError::Store)?;
    if tail_offset > policy.max_ledger_bytes {
        return Err(PublisherError::Store);
    }
    let mut header = Vec::with_capacity(header_len);
    header.extend_from_slice(LEDGER_MAGIC);
    header.extend_from_slice(policy.ledger_domain.as_bytes());
    header.push(b'\n');
    let initial = encode_checkpoint(CommitCheckpoint {
        generation: 0,
        tail_offset,
        tail_hash: ZERO_HASH,
    });
    for _ in 0..CHECKPOINT_COPIES {
        header.extend_from_slice(&initial);
    }
    if header.len() != header_len {
        return Err(PublisherError::Store);
    }
    Ok(header)
}

fn encode_checkpoint(checkpoint: CommitCheckpoint) -> [u8; CHECKPOINT_COPY_BYTES] {
    let mut encoded = [0u8; CHECKPOINT_COPY_BYTES];
    encoded[..8].copy_from_slice(CHECKPOINT_MAGIC);
    encoded[8..12].copy_from_slice(&CHECKPOINT_VERSION.to_be_bytes());
    encoded[16..24].copy_from_slice(&checkpoint.generation.to_be_bytes());
    encoded[24..32].copy_from_slice(&checkpoint.tail_offset.to_be_bytes());
    encoded[32..64].copy_from_slice(&checkpoint.tail_hash);
    let mut digest = Sha256::new();
    digest.update(b"fe2o3-protected-publisher-ledger-checkpoint-v3\0");
    digest.update(&encoded[..64]);
    digest.update(CHECKPOINT_END_MAGIC);
    let checksum: [u8; 32] = digest.finalize().into();
    encoded[64..96].copy_from_slice(&checksum);
    encoded[96..].copy_from_slice(CHECKPOINT_END_MAGIC);
    encoded
}

fn decode_checkpoint(encoded: &[u8]) -> Option<CommitCheckpoint> {
    if encoded.len() != CHECKPOINT_COPY_BYTES
        || &encoded[..8] != CHECKPOINT_MAGIC
        || u32::from_be_bytes(encoded[8..12].try_into().ok()?) != CHECKPOINT_VERSION
        || encoded[12..16] != [0; 4]
        || &encoded[96..] != CHECKPOINT_END_MAGIC
    {
        return None;
    }
    let mut digest = Sha256::new();
    digest.update(b"fe2o3-protected-publisher-ledger-checkpoint-v3\0");
    digest.update(&encoded[..64]);
    digest.update(CHECKPOINT_END_MAGIC);
    let checksum: [u8; 32] = digest.finalize().into();
    if encoded[64..96] != checksum {
        return None;
    }
    Some(CommitCheckpoint {
        generation: u64::from_be_bytes(encoded[16..24].try_into().ok()?),
        tail_offset: u64::from_be_bytes(encoded[24..32].try_into().ok()?),
        tail_hash: encoded[32..64].try_into().ok()?,
    })
}

fn select_checkpoint(
    region: &[u8],
    header_len: u64,
    max_receipts: u64,
    max_ledger_bytes: u64,
) -> Result<CommitCheckpoint, PublisherError> {
    if region.len() != CHECKPOINT_REGION_BYTES {
        return Err(PublisherError::Store);
    }
    let mut valid = Vec::with_capacity(CHECKPOINT_COPIES);
    for encoded in region.chunks_exact(CHECKPOINT_COPY_BYTES) {
        let Some(checkpoint) = decode_checkpoint(encoded) else {
            continue;
        };
        let minimum_tail = header_len
            .checked_add(
                checkpoint
                    .generation
                    .checked_mul(MIN_FRAME_BYTES as u64)
                    .ok_or(PublisherError::Store)?,
            )
            .ok_or(PublisherError::Store)?;
        let maximum_tail = header_len
            .checked_add(
                checkpoint
                    .generation
                    .checked_mul(MAX_LEDGER_FRAME_BYTES as u64)
                    .ok_or(PublisherError::Store)?,
            )
            .ok_or(PublisherError::Store)?;
        if checkpoint.generation > max_receipts
            || checkpoint.tail_offset < minimum_tail
            || checkpoint.tail_offset > maximum_tail
            || checkpoint.tail_offset > max_ledger_bytes
            || (checkpoint.generation == 0
                && (checkpoint.tail_offset != header_len || checkpoint.tail_hash != ZERO_HASH))
            || (checkpoint.generation != 0 && checkpoint.tail_hash == ZERO_HASH)
        {
            return Err(PublisherError::Store);
        }
        if valid.iter().any(|other: &CommitCheckpoint| {
            other.generation == checkpoint.generation && *other != checkpoint
        }) {
            return Err(PublisherError::Store);
        }
        valid.push(checkpoint);
    }
    valid
        .into_iter()
        .max_by_key(|checkpoint| checkpoint.generation)
        .ok_or(PublisherError::Store)
}

fn encode_frame(
    sequence: u64,
    previous_hash: [u8; 32],
    payload: &[u8],
) -> Result<Vec<u8>, PublisherError> {
    checked_frame_length(payload.len())?;
    let payload_length = u32::try_from(payload.len()).map_err(|_| PublisherError::Store)?;
    let mut prefix = Vec::with_capacity(FRAME_PREFIX_BYTES);
    prefix.extend_from_slice(FRAME_MAGIC);
    prefix.extend_from_slice(&FRAME_VERSION.to_be_bytes());
    prefix.extend_from_slice(&payload_length.to_be_bytes());
    prefix.extend_from_slice(&sequence.to_be_bytes());
    prefix.extend_from_slice(&previous_hash);
    let hash = frame_hash(&prefix, payload);
    let frame_length = checked_frame_length(payload.len())? as u64;
    let trailer = frame_trailer(&prefix, payload, hash, frame_length, sequence);
    let mut frame = Vec::with_capacity(frame_length as usize);
    frame.extend_from_slice(&prefix);
    frame.extend_from_slice(payload);
    frame.extend_from_slice(&hash);
    frame.extend_from_slice(&trailer);
    Ok(frame)
}

fn checked_frame_length(payload_length: usize) -> Result<usize, PublisherError> {
    if payload_length == 0 || payload_length > MAX_LEDGER_RECORD_BYTES {
        return Err(PublisherError::Store);
    }
    let frame_length = FRAME_PREFIX_BYTES
        .checked_add(payload_length)
        .and_then(|value| value.checked_add(FRAME_HASH_BYTES))
        .and_then(|value| value.checked_add(FRAME_TRAILER_BYTES))
        .ok_or(PublisherError::Store)?;
    if frame_length > MAX_LEDGER_FRAME_BYTES {
        return Err(PublisherError::Store);
    }
    Ok(frame_length)
}

fn frame_length_from_prefix(
    prefix: &[u8; FRAME_PREFIX_BYTES],
    expected_sequence: u64,
    expected_previous_hash: [u8; 32],
) -> Result<usize, PublisherError> {
    if &prefix[..8] != FRAME_MAGIC
        || u32::from_be_bytes(prefix[8..12].try_into().unwrap()) != FRAME_VERSION
        || u64::from_be_bytes(prefix[16..24].try_into().unwrap()) != expected_sequence
        || <[u8; 32]>::try_from(&prefix[24..56]).unwrap() != expected_previous_hash
    {
        return Err(PublisherError::Store);
    }
    checked_frame_length(u32::from_be_bytes(prefix[12..16].try_into().unwrap()) as usize)
}

fn decode_frame(
    frame: &[u8],
    expected_sequence: u64,
    expected_previous_hash: [u8; 32],
) -> Result<DecodedFrame, PublisherError> {
    if frame.len() < MIN_FRAME_BYTES {
        return Err(PublisherError::Store);
    }
    let prefix: &[u8; FRAME_PREFIX_BYTES] = frame[..FRAME_PREFIX_BYTES]
        .try_into()
        .map_err(|_| PublisherError::Store)?;
    let frame_length = frame_length_from_prefix(prefix, expected_sequence, expected_previous_hash)?;
    if frame.len() != frame_length {
        return Err(PublisherError::Store);
    }
    let payload_end = frame_length
        .checked_sub(FRAME_HASH_BYTES + FRAME_TRAILER_BYTES)
        .ok_or(PublisherError::Store)?;
    let payload = &frame[FRAME_PREFIX_BYTES..payload_end];
    let computed_hash = frame_hash(prefix, payload);
    let hash_end = payload_end + FRAME_HASH_BYTES;
    if frame[payload_end..hash_end] != computed_hash {
        return Err(PublisherError::Store);
    }
    let trailer = &frame[hash_end..];
    if trailer
        != frame_trailer(
            prefix,
            payload,
            computed_hash,
            frame_length as u64,
            expected_sequence,
        )
    {
        return Err(PublisherError::Store);
    }
    Ok(DecodedFrame {
        frame_hash: computed_hash,
        frame_length: frame_length as u64,
        record: decode_record(payload)?,
    })
}

fn frame_hash(prefix: &[u8], payload: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"fe2o3-protected-publisher-ledger-frame-v3\0");
    hash.update(prefix);
    hash.update(payload);
    hash.finalize().into()
}

fn frame_trailer(
    prefix: &[u8],
    payload: &[u8],
    frame_hash: [u8; 32],
    frame_length: u64,
    sequence: u64,
) -> [u8; FRAME_TRAILER_BYTES] {
    let inverse_length = !frame_length;
    let mut commit = Sha256::new();
    commit.update(b"fe2o3-protected-publisher-ledger-commit-v3\0");
    commit.update(prefix);
    commit.update(payload);
    commit.update(frame_hash);
    commit.update(frame_length.to_be_bytes());
    commit.update(inverse_length.to_be_bytes());
    commit.update(sequence.to_be_bytes());
    let commit_hash: [u8; 32] = commit.finalize().into();

    let mut trailer = [0u8; FRAME_TRAILER_BYTES];
    trailer[..8].copy_from_slice(&frame_length.to_be_bytes());
    trailer[8..16].copy_from_slice(&inverse_length.to_be_bytes());
    trailer[16..24].copy_from_slice(&sequence.to_be_bytes());
    trailer[24..56].copy_from_slice(&frame_hash);
    trailer[56..88].copy_from_slice(&commit_hash);
    trailer[88..].copy_from_slice(FRAME_TRAILER_MAGIC);
    trailer
}

fn decode_record(payload: &[u8]) -> Result<LedgerRecord, PublisherError> {
    let value = parse_canonical_with_string_limit(
        payload,
        MAX_LEDGER_RECORD_BYTES,
        MAX_LEDGER_RESPONSE_BASE64_BYTES,
    )
    .map_err(|_| PublisherError::Store)?;
    let record: LedgerRecord = serde_json::from_value(value).map_err(|_| PublisherError::Store)?;
    validate_record(&record)?;
    Ok(record)
}

fn validate_issue_input(input: &IssueInput<'_>) -> Result<(), PublisherError> {
    if !is_digest(input.request_key_sha256)
        || !is_digest(input.stable_authorization_sha256)
        || !is_digest(input.request_identity)
        || !is_digest(input.request_sha256)
        || input.request_body.len() > MAX_LEDGER_REQUEST_BODY_BYTES
        || request_identity(input.request_body) != input.request_identity
        || raw_request_sha256(input.request_body) != input.request_sha256
    {
        return Err(PublisherError::Store);
    }
    let request_value = parse_canonical(input.request_body, MAX_LEDGER_REQUEST_BODY_BYTES)
        .map_err(|_| PublisherError::Store)?;
    let parsed: PublisherRequest =
        serde_json::from_value(request_value).map_err(|_| PublisherError::Store)?;
    let parsed_bytes =
        canonical_bytes(&serde_json::to_value(&parsed).map_err(|_| PublisherError::Store)?)
            .map_err(|_| PublisherError::Store)?;
    let projection =
        canonical_bytes(&parsed.oidc_authorization).map_err(|_| PublisherError::Store)?;
    if parsed_bytes != input.request_body
        || sha256_hex(&projection) != input.stable_authorization_sha256
    {
        return Err(PublisherError::Store);
    }
    Ok(())
}

fn validate_record(record: &LedgerRecord) -> Result<(), PublisherError> {
    if record.schema_version != 1
        || record.record_domain != "fe2o3-protected-publisher-ledger-record-v1"
        || record.issued_at <= 0
        || !is_digest(&record.evidence_identity)
        || !is_digest(&record.request_identity)
        || !is_digest(&record.request_key_sha256)
        || !is_digest(&record.request_sha256)
        || !is_digest(&record.stable_authorization_sha256)
        || record.request_body_base64.len() > MAX_LEDGER_REQUEST_BASE64_BYTES
        || record.response_body_base64.len() > MAX_LEDGER_RESPONSE_BASE64_BYTES
    {
        return Err(PublisherError::Store);
    }
    let request = decode_base64(
        &record.request_body_base64,
        MAX_LEDGER_REQUEST_BASE64_BYTES,
        MAX_LEDGER_REQUEST_BODY_BYTES,
    )?;
    let response = decode_base64(
        &record.response_body_base64,
        MAX_LEDGER_RESPONSE_BASE64_BYTES,
        MAX_LEDGER_RESPONSE_BODY_BYTES,
    )?;
    if request_identity(&request) != record.request_identity
        || raw_request_sha256(&request) != record.request_sha256
    {
        return Err(PublisherError::Store);
    }
    let request_value = parse_canonical(&request, MAX_LEDGER_REQUEST_BODY_BYTES)
        .map_err(|_| PublisherError::Store)?;
    let request: PublisherRequest =
        serde_json::from_value(request_value).map_err(|_| PublisherError::Store)?;
    let projection =
        canonical_bytes(&request.oidc_authorization).map_err(|_| PublisherError::Store)?;
    if sha256_hex(&projection) != record.stable_authorization_sha256 {
        return Err(PublisherError::Store);
    }
    let response_value = parse_canonical_with_string_limit(
        &response,
        MAX_LEDGER_RESPONSE_BODY_BYTES,
        MAX_RECEIPT_BASE64_BYTES,
    )
    .map_err(|_| PublisherError::Store)?;
    let response_object = response_value.as_object().ok_or(PublisherError::Store)?;
    if response_object.len() != 4
        || response_object
            .get("schema_version")
            .and_then(Value::as_u64)
            != Some(1)
        || response_object
            .get("request_sha256")
            .and_then(Value::as_str)
            != Some(&record.request_sha256)
    {
        return Err(PublisherError::Store);
    }
    let receipt = response_object
        .get("publisher_receipt_base64")
        .and_then(Value::as_str)
        .ok_or(PublisherError::Store)?;
    let receipt = decode_base64(receipt, MAX_RECEIPT_BASE64_BYTES, MAX_RECEIPT_BYTES)?;
    if domain_hash(
        b"fe2o3-protected-publisher-evidence-identity-v1\0",
        &receipt,
    ) != record.evidence_identity
    {
        return Err(PublisherError::Store);
    }
    Ok(())
}

fn decode_base64(
    value: &str,
    encoded_limit: usize,
    decoded_limit: usize,
) -> Result<Vec<u8>, PublisherError> {
    if value.len() > encoded_limit {
        return Err(PublisherError::Store);
    }
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|_| PublisherError::Store)?;
    if decoded.len() > decoded_limit
        || base64::engine::general_purpose::STANDARD.encode(&decoded) != value
    {
        return Err(PublisherError::Store);
    }
    Ok(decoded)
}

fn domain_hash(domain: &[u8], value: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(value);
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256_hex(value: &[u8]) -> String {
    Sha256::digest(value)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn check_deadline(deadline: Instant) -> Result<(), PublisherError> {
    if Instant::now() >= deadline {
        Err(PublisherError::Store)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{Permissions, hard_link};
    use std::os::unix::fs::PermissionsExt;
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::*;
    use crate::receipt::TestSigner;
    use crate::test_support::{Fixture, fixture, secure_tempdir};

    const KEY_A: &str = "71a1de4805f764bdf13f374906476fbc60d23f0e4f93f6d63c33f2c4029d6605";
    const KEY_B: &str = "88578b21c1dbb86eace7b852723ab32b7564856518468168165f75890dd14b8e";
    const KEY_C: &str = "9c3d7e4182a605bf104ed8793a47cbf62169d508eacc638f725b1df4803ae796";

    fn key_digest(key: &str) -> String {
        domain_hash(
            b"fe2o3-protected-publisher-idempotency-key-v1\0",
            key.as_bytes(),
        )
    }

    fn projection_digest(request: &PublisherRequest) -> String {
        sha256_hex(&canonical_bytes(&request.oidc_authorization).unwrap())
    }

    fn issue_with(
        store: &mut DurableStore,
        key: &str,
        fixture: &crate::test_support::Fixture,
    ) -> Result<Vec<u8>, PublisherError> {
        let signer = TestSigner::new("test-publisher-v1");
        let identity = request_identity(&fixture.request_body);
        let request_sha = raw_request_sha256(&fixture.request_body);
        let auth = projection_digest(&fixture.request);
        store.issue(IssueInput {
            request_key_sha256: &key_digest(key),
            stable_authorization_sha256: &auth,
            request_identity: &identity,
            request_sha256: &request_sha,
            request_body: &fixture.request_body,
            request: &fixture.request,
            issued_at: 1_800_000_000,
            signature_domain: "test",
            signer: &signer,
        })
    }

    fn distinct_fixture() -> Fixture {
        let mut value = fixture();
        value.request.archive_sha256 = "e".repeat(64);
        value.request_body =
            canonical_bytes(&serde_json::to_value(&value.request).unwrap()).unwrap();
        value
    }

    fn third_fixture() -> Fixture {
        let mut value = fixture();
        value.request.archive_sha256 = "d".repeat(64);
        value.request_body =
            canonical_bytes(&serde_json::to_value(&value.request).unwrap()).unwrap();
        value
    }

    fn assert_poisoned_lookup_failure(path: &Path, configure: impl FnOnce(&mut DurableStore)) {
        let first = fixture();
        let second = distinct_fixture();
        let mut store = DurableStore::open(path).unwrap();
        let response = issue_with(&mut store, KEY_A, &first).unwrap();
        let durable_before = std::fs::read(path).unwrap();
        store.file.seek(SeekFrom::Start(17)).unwrap();
        configure(&mut store);

        assert!(matches!(
            issue_with(&mut store, KEY_A, &first),
            Err(PublisherError::Store)
        ));
        assert_eq!(store.file.stream_position().unwrap(), 17);
        assert!(store.poisoned);
        assert!(matches!(
            issue_with(&mut store, KEY_B, &second),
            Err(PublisherError::Store)
        ));
        assert_eq!(std::fs::read(path).unwrap(), durable_before);
        drop(store);

        let mut reopened = DurableStore::open(path).unwrap();
        assert_eq!(reopened.count(), 1);
        assert_eq!(issue_with(&mut reopened, KEY_A, &first).unwrap(), response);
    }

    fn rewrite_local_frame_record(frame: &mut [u8], mutate: impl FnOnce(&mut LedgerRecord)) {
        let payload_end = frame.len() - FRAME_HASH_BYTES - FRAME_TRAILER_BYTES;
        let mut record = decode_record(&frame[FRAME_PREFIX_BYTES..payload_end]).unwrap();
        mutate(&mut record);
        let payload = canonical_bytes(&serde_json::to_value(record).unwrap()).unwrap();
        assert_eq!(payload.len(), payload_end - FRAME_PREFIX_BYTES);
        frame[FRAME_PREFIX_BYTES..payload_end].copy_from_slice(&payload);
        let hash = frame_hash(&frame[..FRAME_PREFIX_BYTES], &payload);
        frame[payload_end..payload_end + FRAME_HASH_BYTES].copy_from_slice(&hash);
        let sequence = u64::from_be_bytes(frame[16..24].try_into().unwrap());
        let trailer = frame_trailer(
            &frame[..FRAME_PREFIX_BYTES],
            &payload,
            hash,
            frame.len() as u64,
            sequence,
        );
        frame[payload_end + FRAME_HASH_BYTES..].copy_from_slice(&trailer);
    }

    fn write_ledger(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).unwrap();
        std::fs::set_permissions(path, Permissions::from_mode(0o600)).unwrap();
    }

    fn assert_restart_rejects_unchanged(path: &Path, bytes: &[u8], label: &str) {
        write_ledger(path, bytes);
        assert!(DurableStore::open(path).is_err(), "accepted {label}");
        assert_eq!(std::fs::read(path).unwrap(), bytes, "modified {label}");
    }

    fn frame_boundaries(store: &DurableStore) -> Vec<(usize, usize)> {
        let mut entries = store.by_request_key.values().collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.sequence);
        entries
            .into_iter()
            .map(|entry| (entry.frame_offset as usize, entry.frame_length as usize))
            .collect()
    }

    fn fixture_with_reference(reference: &str) -> Fixture {
        let mut fixture = fixture();
        let caller = format!("powderluv/fe2o3/.github/workflows/parity-promotion.yml@{reference}");
        let protected =
            format!("powderluv/fe2o3/.github/workflows/parity-publisher-gate.yml@{reference}");
        fixture.request.oidc_authorization["ref"] = Value::String(reference.into());
        fixture.request.oidc_authorization["workflow_ref"] = Value::String(caller.clone());
        fixture.request.oidc_authorization["job_workflow_ref"] = Value::String(protected.clone());
        fixture
            .request
            .workflow
            .insert("github_ref".into(), reference.into());
        fixture
            .request
            .workflow
            .insert("github_workflow_ref".into(), caller);
        fixture.request_body =
            canonical_bytes(&serde_json::to_value(&fixture.request).unwrap()).unwrap();
        fixture
    }

    fn reference_with_length(length: usize) -> String {
        let prefix = "refs/heads/gh-readonly-queue/main/";
        assert!(length > prefix.len());
        format!("{prefix}pr-{}", "x".repeat(length - prefix.len() - 3))
    }

    #[test]
    fn exact_retry_returns_identical_durable_bytes() {
        let temp = secure_tempdir();
        let path = temp.path().join("publisher.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        let fixture = fixture();
        let first = issue_with(&mut store, KEY_A, &fixture).unwrap();
        let second = issue_with(&mut store, KEY_A, &fixture).unwrap();
        assert_eq!(first, second);
        assert_eq!(store.count(), 1);
        drop(store);
        let mut reopened = DurableStore::open(&path).unwrap();
        assert_eq!(issue_with(&mut reopened, KEY_A, &fixture).unwrap(), first);
    }

    #[test]
    fn indexed_retry_is_observational_under_short_reads_and_eintr() {
        let temp = secure_tempdir();
        let path = temp.path().join("publisher.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        let first = fixture();
        let response = issue_with(&mut store, KEY_A, &first).unwrap();
        let sequence = store.next_sequence;
        let hash = store.tail_hash;
        let tail = store.tail_offset;
        let index = store.by_request_key.clone();
        store.file.seek(SeekFrom::Start(19)).unwrap();
        store.set_maximum_indexed_read_chunk(1);
        store.interrupt_indexed_reads(MAX_INDEXED_READ_INTERRUPTS);

        assert_eq!(issue_with(&mut store, KEY_A, &first).unwrap(), response);
        assert_eq!(store.next_sequence, sequence);
        assert_eq!(store.tail_hash, hash);
        assert_eq!(store.tail_offset, tail);
        assert_eq!(store.by_request_key.len(), index.len());
        for (key, expected) in index {
            let actual = store.by_request_key.get(&key).unwrap();
            assert_eq!(actual.frame_offset, expected.frame_offset);
            assert_eq!(actual.frame_length, expected.frame_length);
            assert_eq!(actual.sequence, expected.sequence);
            assert_eq!(actual.previous_hash, expected.previous_hash);
            assert_eq!(actual.frame_hash, expected.frame_hash);
        }
        assert_eq!(store.file.stream_position().unwrap(), 19);

        issue_with(&mut store, KEY_B, &distinct_fixture()).unwrap();
        drop(store);
        assert_eq!(DurableStore::open(&path).unwrap().count(), 2);
    }

    #[test]
    fn indexed_read_call_eof_and_eintr_failures_poison_without_append() {
        for (name, configure) in [
            (
                "first-read-eio",
                DurableStore::fail_indexed_read_on_call as fn(&mut DurableStore, usize),
            ),
            (
                "second-read-eio",
                DurableStore::fail_indexed_read_on_call as fn(&mut DurableStore, usize),
            ),
            (
                "first-read-eof",
                DurableStore::eof_indexed_read_on_call as fn(&mut DurableStore, usize),
            ),
            (
                "second-read-eof",
                DurableStore::eof_indexed_read_on_call as fn(&mut DurableStore, usize),
            ),
        ] {
            let temp = secure_tempdir();
            let path = temp.path().join(format!("{name}.ledger"));
            let call = if name.starts_with("first") { 1 } else { 2 };
            assert_poisoned_lookup_failure(&path, |store| configure(store, call));
        }

        let temp = secure_tempdir();
        let path = temp.path().join("eintr-exhaustion.ledger");
        assert_poisoned_lookup_failure(&path, |store| {
            store.interrupt_indexed_reads(MAX_INDEXED_READ_INTERRUPTS + 1);
        });
    }

    #[test]
    fn every_indexed_partial_read_boundary_poison_fails_closed() {
        let temp = secure_tempdir();
        let path = temp.path().join("partial.ledger");
        let first = fixture();
        let second = distinct_fixture();
        let mut store = DurableStore::open(&path).unwrap();
        issue_with(&mut store, KEY_A, &first).unwrap();
        let frame_length = store
            .by_request_key
            .get(&key_digest(KEY_A))
            .unwrap()
            .frame_length as usize;
        let durable = std::fs::read(&path).unwrap();

        for boundary in 0..frame_length {
            store.fail_indexed_read_after(boundary);
            assert!(matches!(
                issue_with(&mut store, KEY_A, &first),
                Err(PublisherError::Store)
            ));
            assert!(matches!(
                issue_with(&mut store, KEY_B, &second),
                Err(PublisherError::Store)
            ));
            assert_eq!(
                std::fs::read(&path).unwrap(),
                durable,
                "boundary {boundary}"
            );
            drop(store);
            store = DurableStore::open(&path).unwrap();
            assert_eq!(store.count(), 1);
        }
    }

    #[test]
    fn malformed_local_frames_and_index_mismatches_poison_without_disk_mutation() {
        type FrameMutation = fn(&mut Vec<u8>);
        let cases: [(&str, FrameMutation); 4] = [
            ("canonical", |frame| {
                let payload_end = frame.len() - FRAME_HASH_BYTES - FRAME_TRAILER_BYTES;
                frame[FRAME_PREFIX_BYTES] = b'!';
                let hash = frame_hash(
                    &frame[..FRAME_PREFIX_BYTES],
                    &frame[FRAME_PREFIX_BYTES..payload_end],
                );
                frame[payload_end..payload_end + FRAME_HASH_BYTES].copy_from_slice(&hash);
                let sequence = u64::from_be_bytes(frame[16..24].try_into().unwrap());
                let trailer = frame_trailer(
                    &frame[..FRAME_PREFIX_BYTES],
                    &frame[FRAME_PREFIX_BYTES..payload_end],
                    hash,
                    frame.len() as u64,
                    sequence,
                );
                frame[payload_end + FRAME_HASH_BYTES..].copy_from_slice(&trailer);
            }),
            ("base64", |frame| {
                rewrite_local_frame_record(frame, |record| {
                    record.request_body_base64.replace_range(..1, "!");
                });
            }),
            ("digest", |frame| {
                rewrite_local_frame_record(frame, |record| {
                    let replacement = if record.request_sha256.starts_with('0') {
                        "1"
                    } else {
                        "0"
                    };
                    record.request_sha256.replace_range(..1, replacement);
                });
            }),
            ("semantic", |frame| {
                rewrite_local_frame_record(frame, |record| {
                    assert!(record.record_domain.pop().is_some());
                    record.record_domain.push('2');
                });
            }),
        ];
        for (name, mutate) in cases {
            let temp = secure_tempdir();
            let path = temp.path().join(format!("{name}.ledger"));
            assert_poisoned_lookup_failure(&path, |store| {
                store.set_before_indexed_decode(mutate);
            });
        }

        for name in [
            "offset",
            "length",
            "sequence",
            "previous-hash",
            "frame-hash",
            "request-key",
            "request-identity",
            "request-digest",
            "stable-authorization",
        ] {
            let temp = secure_tempdir();
            let path = temp.path().join(format!("index-{name}.ledger"));
            assert_poisoned_lookup_failure(&path, |store| {
                let index = store.by_request_key.get_mut(&key_digest(KEY_A)).unwrap();
                match name {
                    "offset" => index.frame_offset += 1,
                    "length" => index.frame_length -= 1,
                    "sequence" => index.sequence += 1,
                    "previous-hash" => index.previous_hash[0] ^= 1,
                    "frame-hash" => index.frame_hash[0] ^= 1,
                    "request-key" => {
                        let replacement = if index.request_key_sha256.starts_with('0') {
                            "1"
                        } else {
                            "0"
                        };
                        index.request_key_sha256.replace_range(..1, replacement);
                    }
                    "request-identity" => {
                        let replacement = if index.request_identity.starts_with('0') {
                            "1"
                        } else {
                            "0"
                        };
                        index.request_identity.replace_range(..1, replacement);
                    }
                    "request-digest" => {
                        let replacement = if index.request_sha256.starts_with('0') {
                            "1"
                        } else {
                            "0"
                        };
                        index.request_sha256.replace_range(..1, replacement);
                    }
                    "stable-authorization" => {
                        let replacement = if index.stable_authorization_sha256.starts_with('0') {
                            "1"
                        } else {
                            "0"
                        };
                        index
                            .stable_authorization_sha256
                            .replace_range(..1, replacement);
                    }
                    _ => unreachable!(),
                }
            });
        }
    }

    #[test]
    fn hostile_prechecked_index_substitution_bypasses_poison() {
        for name in ["request-identity", "request-digest", "stable-authorization"] {
            let temp = secure_tempdir();
            let path = temp.path().join(format!("hostile-precheck-{name}.ledger"));
            let first = fixture();
            let second = distinct_fixture();
            let mut store = DurableStore::open(&path).unwrap();
            issue_with(&mut store, KEY_A, &first).unwrap();
            let durable = std::fs::read(&path).unwrap();
            let index = store.by_request_key.get_mut(&key_digest(KEY_A)).unwrap();
            let field = match name {
                "request-identity" => &mut index.request_identity,
                "request-digest" => &mut index.request_sha256,
                "stable-authorization" => &mut index.stable_authorization_sha256,
                _ => unreachable!(),
            };
            let replacement = if field.starts_with('0') { "1" } else { "0" };
            field.replace_range(..1, replacement);

            assert!(matches!(
                issue_with(&mut store, KEY_A, &first),
                Err(PublisherError::Store)
            ));
            assert!(store.poisoned, "{name} did not poison");
            assert!(matches!(
                issue_with(&mut store, KEY_A, &first),
                Err(PublisherError::Store)
            ));
            assert!(matches!(
                issue_with(&mut store, KEY_B, &second),
                Err(PublisherError::Store)
            ));
            assert_eq!(std::fs::read(&path).unwrap(), durable);
            drop(store);
            assert_eq!(DurableStore::open(&path).unwrap().count(), 1);
        }
    }

    #[test]
    fn serialized_concurrent_exact_retries_remain_idempotent() {
        let temp = secure_tempdir();
        let path = temp.path().join("concurrent-retries.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        let expected = issue_with(&mut store, KEY_A, &fixture()).unwrap();
        let store = Arc::new(std::sync::Mutex::new(store));
        let start = Arc::new(Barrier::new(16));
        let threads = (0..16)
            .map(|_| {
                let store = store.clone();
                let start = start.clone();
                let expected = expected.clone();
                thread::spawn(move || {
                    start.wait();
                    let actual = issue_with(&mut store.lock().unwrap(), KEY_A, &fixture()).unwrap();
                    assert_eq!(actual, expected);
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
        let store = match Arc::try_unwrap(store) {
            Ok(store) => store.into_inner().unwrap(),
            Err(_) => panic!("retry workers retained the store"),
        };
        assert_eq!(store.count(), 1);
        drop(store);
        assert_eq!(DurableStore::open(&path).unwrap().count(), 1);
    }

    #[test]
    #[ignore = "external LD_PRELOAD control proving indexed retry uses no lseek"]
    fn hostile_transient_index_seek_error_must_not_corrupt_later_append_state() {
        let temp = secure_tempdir();
        let path = temp.path().join("publisher.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        let first = fixture();
        let response = issue_with(&mut store, KEY_A, &first).unwrap();

        assert_eq!(issue_with(&mut store, KEY_A, &first).unwrap(), response);
        issue_with(&mut store, KEY_B, &distinct_fixture()).unwrap();
        drop(store);
        assert_eq!(DurableStore::open(&path).unwrap().count(), 2);
    }

    #[test]
    fn preexisting_empty_and_partial_headers_fail_closed_unchanged() {
        let temp = secure_tempdir();
        let header = ledger_header(&StorePolicy::test_default()).unwrap();
        for prefix in 0..header.len() {
            let path = temp.path().join(format!("legacy-prefix-{prefix}.ledger"));
            std::fs::write(&path, &header[..prefix]).unwrap();
            std::fs::set_permissions(&path, Permissions::from_mode(0o600)).unwrap();
            assert!(DurableStore::open(&path).is_err(), "prefix {prefix}");
            assert_eq!(std::fs::read(&path).unwrap(), &header[..prefix]);
        }
    }

    #[test]
    fn legacy_v1_and_v2_ledgers_are_rejected_without_migration_or_mutation() {
        let temp = secure_tempdir();
        let policy = StorePolicy::test_default();
        for (version, magic) in LEGACY_LEDGER_MAGICS.iter().enumerate() {
            let mut legacy = Vec::new();
            legacy.extend_from_slice(magic);
            legacy.extend_from_slice(policy.ledger_domain.as_bytes());
            legacy.push(b'\n');
            legacy.extend_from_slice(b"complete-legacy-ledger-bytes-must-not-be-reinterpreted");
            let path = temp.path().join(format!("legacy-v{}.ledger", version + 1));
            std::fs::write(&path, &legacy).unwrap();
            std::fs::set_permissions(&path, Permissions::from_mode(0o600)).unwrap();

            assert!(DurableStore::open_with_policy(&path, policy.clone()).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), legacy);
        }
    }

    #[test]
    fn maximum_merge_queue_ref_commits_and_restarts() {
        let temp = secure_tempdir();
        let path = temp.path().join("maximum-ref.ledger");
        let reference = reference_with_length(234);
        let fixture = fixture_with_reference(&reference);
        let encoded = base64::engine::general_purpose::STANDARD.encode(&fixture.request_body);
        assert_eq!(reference.len(), 234);
        assert!((3_500..=3_900).contains(&fixture.request_body.len()));
        assert!((4_600..=5_200).contains(&encoded.len()));
        assert!(encoded.len() > crate::bounds::MAX_JSON_STRING_BYTES);

        let mut store = DurableStore::open(&path).unwrap();
        let response = issue_with(&mut store, KEY_A, &fixture).unwrap();
        drop(store);
        let mut reopened = DurableStore::open(&path).unwrap();
        assert_eq!(
            issue_with(&mut reopened, KEY_A, &fixture).unwrap(),
            response
        );
    }

    #[test]
    fn accepted_request_corpus_is_restart_decodable() {
        let temp = secure_tempdir();
        let prefix = "refs/heads/gh-readonly-queue/main/";
        let references = [
            format!("{prefix}pr-1"),
            reference_with_length(64),
            reference_with_length(128),
            reference_with_length(234),
            format!("{prefix}pr-\u{e9}"),
        ];
        for (index, reference) in references.iter().enumerate() {
            let path = temp.path().join(format!("corpus-{index}.ledger"));
            let fixture = fixture_with_reference(reference);
            let mut store = DurableStore::open(&path).unwrap();
            let response = issue_with(&mut store, KEY_A, &fixture).unwrap();
            drop(store);
            let mut reopened = DurableStore::open(&path).unwrap();
            assert_eq!(
                issue_with(&mut reopened, KEY_A, &fixture).unwrap(),
                response
            );
        }
    }

    #[test]
    fn decoded_base64_and_frame_bounds_are_exact_and_overflow_safe() {
        for (decoded_limit, encoded_limit) in [
            (
                MAX_LEDGER_REQUEST_BODY_BYTES,
                MAX_LEDGER_REQUEST_BASE64_BYTES,
            ),
            (
                MAX_LEDGER_RESPONSE_BODY_BYTES,
                MAX_LEDGER_RESPONSE_BASE64_BYTES,
            ),
            (MAX_RECEIPT_BYTES, MAX_RECEIPT_BASE64_BYTES),
        ] {
            let exact = vec![0u8; decoded_limit];
            let exact_encoded = base64::engine::general_purpose::STANDARD.encode(&exact);
            assert_eq!(exact_encoded.len(), encoded_limit);
            assert_eq!(
                decode_base64(&exact_encoded, encoded_limit, decoded_limit).unwrap(),
                exact
            );

            let over = vec![0u8; decoded_limit + 1];
            let over_encoded = base64::engine::general_purpose::STANDARD.encode(over);
            assert!(decode_base64(&over_encoded, encoded_limit, decoded_limit).is_err());
            let mut encoded_over = exact_encoded;
            encoded_over.push('A');
            assert!(decode_base64(&encoded_over, encoded_limit, decoded_limit).is_err());
        }

        assert_eq!(
            checked_frame_length(MAX_LEDGER_RECORD_BYTES).unwrap(),
            MAX_LEDGER_FRAME_BYTES
        );
        assert!(checked_frame_length(MAX_LEDGER_RECORD_BYTES + 1).is_err());
        assert!(checked_frame_length(usize::MAX).is_err());
        assert!(encode_frame(1, ZERO_HASH, &vec![0; MAX_LEDGER_RECORD_BYTES]).is_ok());
        assert!(encode_frame(1, ZERO_HASH, &vec![0; MAX_LEDGER_RECORD_BYTES + 1]).is_err());
    }

    #[test]
    fn restart_rejected_frame_is_never_written_or_acknowledged() {
        let temp = secure_tempdir();
        let path = temp.path().join("preflight.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        let initial_length = std::fs::metadata(&path).unwrap().len();
        store.set_before_admission(|frame| {
            let payload_byte = FRAME_PREFIX_BYTES + 1;
            frame[payload_byte] ^= 1;
        });
        assert!(matches!(
            issue_with(&mut store, KEY_A, &fixture()),
            Err(PublisherError::Store)
        ));
        assert_eq!(store.count(), 0);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), initial_length);
        drop(store);
        assert_eq!(DurableStore::open(&path).unwrap().count(), 0);
    }

    #[test]
    fn row_and_byte_capacity_stop_new_issuance() {
        let temp = secure_tempdir();
        let row_path = temp.path().join("row-capacity.ledger");
        let mut policy = StorePolicy::test_default();
        policy.max_receipts = 1;
        let mut store = DurableStore::open_with_policy(&row_path, policy).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        let mut second = fixture();
        second.request.archive_sha256 = "e".repeat(64);
        second.request_body =
            canonical_bytes(&serde_json::to_value(&second.request).unwrap()).unwrap();
        assert!(matches!(
            issue_with(&mut store, KEY_B, &second),
            Err(PublisherError::Store)
        ));

        let byte_path = temp.path().join("byte-capacity.ledger");
        let mut store = DurableStore::open(&byte_path).unwrap();
        store.policy.max_ledger_bytes = store.tail_offset + 1;
        assert!(matches!(
            issue_with(&mut store, KEY_A, &fixture()),
            Err(PublisherError::Store)
        ));
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn key_body_authorization_and_request_substitution_fail_closed() {
        let temp = secure_tempdir();
        let path = temp.path().join("publisher.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        let first = fixture();
        let response = issue_with(&mut store, KEY_A, &first).unwrap();

        let mut changed_body = fixture();
        changed_body.request.archive_sha256 = "e".repeat(64);
        changed_body.request_body =
            canonical_bytes(&serde_json::to_value(&changed_body.request).unwrap()).unwrap();
        assert!(matches!(
            issue_with(&mut store, KEY_A, &changed_body),
            Err(PublisherError::ReplayConflict)
        ));
        assert!(!store.poisoned);
        assert_eq!(issue_with(&mut store, KEY_A, &first).unwrap(), response);
        assert!(matches!(
            issue_with(&mut store, KEY_B, &first),
            Err(PublisherError::ReplayConflict)
        ));
        assert!(!store.poisoned);

        let mut changed_auth = fixture();
        changed_auth.request.oidc_authorization["actor_id"] = Value::String("202".into());
        changed_auth.request_body =
            canonical_bytes(&serde_json::to_value(&changed_auth.request).unwrap()).unwrap();
        assert!(matches!(
            issue_with(&mut store, KEY_A, &changed_auth),
            Err(PublisherError::ReplayConflict)
        ));
        assert!(!store.poisoned);
        assert_eq!(issue_with(&mut store, KEY_A, &first).unwrap(), response);
        issue_with(&mut store, KEY_B, &changed_body).unwrap();
        drop(store);
        assert_eq!(DurableStore::open(&path).unwrap().count(), 2);
    }

    #[test]
    fn complete_corruption_and_duplicate_frames_reject_on_restart() {
        let temp = secure_tempdir();
        let path = temp.path().join("publisher.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        let header_len = store.header_len;
        drop(store);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[header_len as usize + FRAME_PREFIX_BYTES + 3] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(DurableStore::open(&path).is_err());

        let duplicate_path = temp.path().join("committed-duplicate.ledger");
        let mut store = DurableStore::open(&duplicate_path).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        let first_frame = {
            let bytes = std::fs::read(&duplicate_path).unwrap();
            bytes[store.header_len as usize..].to_vec()
        };
        let first_payload = &first_frame
            [FRAME_PREFIX_BYTES..first_frame.len() - FRAME_HASH_BYTES - FRAME_TRAILER_BYTES];
        let first_hash = store.tail_hash;
        let checkpoint_offset = store.checkpoint_offset as usize;
        let duplicate = encode_frame(2, first_hash, first_payload).unwrap();
        drop(store);
        let mut committed = std::fs::read(&duplicate_path).unwrap();
        committed.extend_from_slice(&duplicate);
        let duplicate_hash = decode_frame(&duplicate, 2, first_hash).unwrap().frame_hash;
        let checkpoint = encode_checkpoint(CommitCheckpoint {
            generation: 2,
            tail_offset: committed.len() as u64,
            tail_hash: duplicate_hash,
        });
        for copy in 0..CHECKPOINT_COPIES {
            let start = checkpoint_offset + copy * CHECKPOINT_COPY_BYTES;
            committed[start..start + CHECKPOINT_COPY_BYTES].copy_from_slice(&checkpoint);
        }
        std::fs::write(&duplicate_path, &committed).unwrap();
        assert!(DurableStore::open(&duplicate_path).is_err());
        assert_eq!(std::fs::read(&duplicate_path).unwrap(), committed);
    }

    #[test]
    fn every_complete_frame_byte_mutation_rejects_without_disk_mutation() {
        let temp = secure_tempdir();
        let source = temp.path().join("source.ledger");
        let mut store = DurableStore::open(&source).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        issue_with(&mut store, KEY_B, &distinct_fixture()).unwrap();
        issue_with(&mut store, KEY_C, &third_fixture()).unwrap();
        let boundaries = frame_boundaries(&store);
        drop(store);
        let durable = std::fs::read(&source).unwrap();

        for (position, (frame_offset, frame_length)) in boundaries.iter().copied().enumerate() {
            for frame_index in 0..frame_length {
                let mut mutated = durable.clone();
                mutated[frame_offset + frame_index] ^= 1;
                let path = temp
                    .path()
                    .join(format!("mutation-{position}-{frame_index}.ledger"));
                assert_restart_rejects_unchanged(
                    &path,
                    &mutated,
                    &format!("frame {position} byte {frame_index}"),
                );
            }
        }
    }

    #[test]
    fn every_checkpoint_byte_has_redundant_recognition_or_fails_closed() {
        let temp = secure_tempdir();
        let source = temp.path().join("checkpoint-source.ledger");
        let mut store = DurableStore::open(&source).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        let checkpoint_offset = store.checkpoint_offset as usize;
        drop(store);
        let durable = std::fs::read(&source).unwrap();

        for region_index in 0..CHECKPOINT_REGION_BYTES {
            let mut mutated = durable.clone();
            mutated[checkpoint_offset + region_index] ^= 1;
            let path = temp.path().join(format!("one-copy-{region_index}.ledger"));
            write_ledger(&path, &mutated);
            assert_eq!(
                DurableStore::open(&path).unwrap().count(),
                1,
                "single checkpoint-region byte {region_index}"
            );
            assert_eq!(std::fs::read(&path).unwrap(), mutated);
        }

        for surviving_copy in 0..CHECKPOINT_COPIES {
            for copy_index in 0..CHECKPOINT_COPY_BYTES {
                let mut mutated = durable.clone();
                for copy in 0..CHECKPOINT_COPIES {
                    if copy != surviving_copy {
                        mutated[checkpoint_offset + copy * CHECKPOINT_COPY_BYTES + copy_index] ^= 1;
                    }
                }
                let path = temp
                    .path()
                    .join(format!("two-copy-{surviving_copy}-{copy_index}.ledger"));
                write_ledger(&path, &mutated);
                assert_eq!(
                    DurableStore::open(&path).unwrap().count(),
                    1,
                    "checkpoint byte {copy_index} with copy {surviving_copy} intact"
                );
                assert_eq!(std::fs::read(&path).unwrap(), mutated);
            }
        }

        for copy_index in 0..CHECKPOINT_COPY_BYTES {
            let mut mutated = durable.clone();
            for copy in 0..CHECKPOINT_COPIES {
                mutated[checkpoint_offset + copy * CHECKPOINT_COPY_BYTES + copy_index] ^= 1;
            }
            let path = temp.path().join(format!("all-copies-{copy_index}.ledger"));
            assert_restart_rejects_unchanged(
                &path,
                &mutated,
                &format!("same byte {copy_index} in every checkpoint copy"),
            );
        }
    }

    #[test]
    fn checkpoint_generation_selection_handles_partial_updates_without_ambiguity() {
        let temp = secure_tempdir();
        let source = temp.path().join("checkpoint-generations.ledger");
        let mut store = DurableStore::open(&source).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        let checkpoint_offset = store.checkpoint_offset as usize;
        let header_len = store.header_len;
        let committed = CommitCheckpoint {
            generation: store.checkpoint_generation,
            tail_offset: store.tail_offset,
            tail_hash: store.tail_hash,
        };
        drop(store);
        let durable = std::fs::read(&source).unwrap();
        let initial = encode_checkpoint(CommitCheckpoint {
            generation: 0,
            tail_offset: header_len,
            tail_hash: ZERO_HASH,
        });

        for new_copies in 1..CHECKPOINT_COPIES {
            let mut partial_update = durable.clone();
            for copy in new_copies..CHECKPOINT_COPIES {
                let start = checkpoint_offset + copy * CHECKPOINT_COPY_BYTES;
                partial_update[start..start + CHECKPOINT_COPY_BYTES].copy_from_slice(&initial);
            }
            let path = temp.path().join(format!("new-copies-{new_copies}.ledger"));
            write_ledger(&path, &partial_update);
            assert_eq!(DurableStore::open(&path).unwrap().count(), 1);
            assert_eq!(std::fs::read(&path).unwrap(), partial_update);
        }

        let conflicting = encode_checkpoint(CommitCheckpoint {
            tail_offset: committed.tail_offset + 1,
            ..committed
        });
        let mut ambiguous = durable.clone();
        let second = checkpoint_offset + CHECKPOINT_COPY_BYTES;
        ambiguous[second..second + CHECKPOINT_COPY_BYTES].copy_from_slice(&conflicting);
        let path = temp.path().join("same-generation-conflict.ledger");
        assert_restart_rejects_unchanged(&path, &ambiguous, "same-generation checkpoint conflict");
    }

    #[test]
    fn checkpoint_short_write_boundaries_never_lose_an_acknowledged_record() {
        for cut in [
            0, 1, 15, 16, 23, 24, 31, 32, 63, 64, 95, 96, 103, 104, 105, 207, 208, 209, 311,
        ] {
            let temp = secure_tempdir();
            let path = temp.path().join(format!("checkpoint-short-{cut}.ledger"));
            let mut store = DurableStore::open(&path).unwrap();
            let header_len = store.header_len;
            let checkpoint_offset = store.checkpoint_offset as usize;
            store.set_maximum_write_chunk(1);
            store.fail_checkpoint_write_after(cut, libc::ENOSPC);
            assert!(matches!(
                issue_with(&mut store, KEY_A, &fixture()),
                Err(PublisherError::Store)
            ));
            assert!(store.poisoned);
            let interrupted = std::fs::read(&path).unwrap();
            let selected = select_checkpoint(
                &interrupted[checkpoint_offset..header_len as usize],
                header_len,
                StorePolicy::test_default().max_receipts,
                StorePolicy::test_default().max_ledger_bytes,
            )
            .unwrap();
            drop(store);

            let reopened = DurableStore::open(&path).unwrap();
            assert_eq!(reopened.count() as u64, selected.generation, "cut {cut}");
            assert_eq!(reopened.tail_offset, selected.tail_offset, "cut {cut}");
        }
    }

    #[test]
    fn multi_frame_length_and_every_trailer_byte_corruption_fails_unchanged() {
        let temp = secure_tempdir();
        let source = temp.path().join("multi-frame-source.ledger");
        let mut store = DurableStore::open(&source).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        issue_with(&mut store, KEY_B, &distinct_fixture()).unwrap();
        issue_with(&mut store, KEY_C, &third_fixture()).unwrap();
        let boundaries = frame_boundaries(&store);
        drop(store);
        let durable = std::fs::read(&source).unwrap();

        for (position, (frame_offset, frame_length)) in boundaries.iter().copied().enumerate() {
            let length_offset = frame_offset + 12;
            let original = u32::from_be_bytes(
                durable[length_offset..length_offset + 4]
                    .try_into()
                    .unwrap(),
            );
            let trailer_offset = frame_offset + frame_length - FRAME_TRAILER_BYTES;
            for trailer_index in 0..FRAME_TRAILER_BYTES {
                let mut mutated = durable.clone();
                mutated[length_offset..length_offset + 4]
                    .copy_from_slice(&original.checked_add(1).unwrap().to_be_bytes());
                mutated[trailer_offset + trailer_index] ^= 1;
                let path = temp
                    .path()
                    .join(format!("length-trailer-{position}-{trailer_index}.ledger"));
                assert_restart_rejects_unchanged(
                    &path,
                    &mutated,
                    &format!("frame {position} forward length plus trailer {trailer_index}"),
                );
            }
        }
    }

    #[test]
    fn multi_frame_length_extremes_and_false_boundaries_never_trigger_committed_truncation() {
        let temp = secure_tempdir();
        let source = temp.path().join("length-extremes-source.ledger");
        let mut store = DurableStore::open(&source).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        issue_with(&mut store, KEY_B, &distinct_fixture()).unwrap();
        issue_with(&mut store, KEY_C, &third_fixture()).unwrap();
        let boundaries = frame_boundaries(&store);
        let committed_tail = store.tail_offset as usize;
        drop(store);
        let durable = std::fs::read(&source).unwrap();

        for (position, (frame_offset, _)) in boundaries.iter().copied().enumerate() {
            let length_offset = frame_offset + 12;
            let original = u32::from_be_bytes(
                durable[length_offset..length_offset + 4]
                    .try_into()
                    .unwrap(),
            );
            for replacement in [
                0,
                1,
                original - 1,
                original + 1,
                MAX_LEDGER_RECORD_BYTES as u32 + 1,
                u32::MAX,
            ] {
                let mut mutated = durable.clone();
                mutated[length_offset..length_offset + 4]
                    .copy_from_slice(&replacement.to_be_bytes());
                let path = temp
                    .path()
                    .join(format!("length-{position}-{replacement}.ledger"));
                assert_restart_rejects_unchanged(
                    &path,
                    &mutated,
                    &format!("frame {position} length {replacement}"),
                );
            }
        }

        let mut false_boundaries = durable.clone();
        for _ in 0..128 {
            false_boundaries.extend_from_slice(FRAME_TRAILER_MAGIC);
            false_boundaries.extend_from_slice(CHECKPOINT_MAGIC);
            false_boundaries.extend_from_slice(FRAME_MAGIC);
        }
        let path = temp.path().join("false-uncommitted-boundaries.ledger");
        write_ledger(&path, &false_boundaries);
        let reopened = DurableStore::open(&path).unwrap();
        assert_eq!(reopened.count(), 3);
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            committed_tail as u64
        );
    }

    #[test]
    fn upward_length_corruption_cannot_erase_a_committed_record() {
        let temp = secure_tempdir();
        let path = temp.path().join("upward-length.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        let payload_length_offset = store.header_len as usize + 12;
        drop(store);
        let mut mutated = std::fs::read(&path).unwrap();
        let original = u32::from_be_bytes(
            mutated[payload_length_offset..payload_length_offset + 4]
                .try_into()
                .unwrap(),
        );
        mutated[payload_length_offset..payload_length_offset + 4]
            .copy_from_slice(&(original + 1).to_be_bytes());
        std::fs::write(&path, &mutated).unwrap();

        assert!(DurableStore::open(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), mutated);
    }

    #[test]
    fn torn_tail_is_truncated_but_complete_frame_is_replayed() {
        let temp = secure_tempdir();
        let source = temp.path().join("source.ledger");
        let mut store = DurableStore::open(&source).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        let header_len = store.header_len as usize;
        drop(store);
        let bytes = std::fs::read(&source).unwrap();
        let header = ledger_header(&StorePolicy::test_default()).unwrap();
        assert_eq!(header.len(), header_len);
        let frame = &bytes[header_len..];
        for cut in 0..frame.len() {
            let path = temp.path().join(format!("torn-{cut}.ledger"));
            let mut partial = header.clone();
            partial.extend_from_slice(&frame[..cut]);
            std::fs::write(&path, partial).unwrap();
            std::fs::set_permissions(&path, Permissions::from_mode(0o600)).unwrap();
            let reopened = DurableStore::open(&path).unwrap();
            assert_eq!(reopened.count(), 0, "partial boundary {cut}");
            assert_eq!(
                std::fs::metadata(&path).unwrap().len(),
                header_len as u64,
                "partial boundary {cut}"
            );

            let committed_path = temp.path().join(format!("committed-cut-{cut}.ledger"));
            let committed_partial = &bytes[..header_len + cut];
            assert_restart_rejects_unchanged(
                &committed_path,
                committed_partial,
                &format!("committed frame truncation {cut}"),
            );
        }
        let complete = temp.path().join("complete.ledger");
        std::fs::write(&complete, bytes).unwrap();
        std::fs::set_permissions(&complete, Permissions::from_mode(0o600)).unwrap();
        assert_eq!(DurableStore::open(&complete).unwrap().count(), 1);
    }

    #[test]
    fn ledger_size_bound_rejects_before_any_tail_recovery() {
        let temp = secure_tempdir();
        let path = temp.path().join("oversized-uncommitted-tail.ledger");
        let mut policy = StorePolicy::test_default();
        policy.max_ledger_bytes = MIN_LEDGER_BYTES;
        let store = DurableStore::open_with_policy(&path, policy.clone()).unwrap();
        drop(store);
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(policy.max_ledger_bytes + 1).unwrap();
        file.sync_data().unwrap();
        drop(file);
        let oversized = std::fs::read(&path).unwrap();

        assert!(DurableStore::open_with_policy(&path, policy).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), oversized);
    }

    #[test]
    fn short_writes_complete_and_enospc_tail_recovers_without_acknowledgement() {
        let temp = secure_tempdir();
        let short = temp.path().join("short.ledger");
        let mut store = DurableStore::open(&short).unwrap();
        store.set_maximum_write_chunk(7);
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        drop(store);
        assert_eq!(DurableStore::open(&short).unwrap().count(), 1);

        let full = temp.path().join("full.ledger");
        let mut store = DurableStore::open(&full).unwrap();
        store.set_maximum_write_chunk(11);
        store.fail_write_after(33, libc::ENOSPC);
        assert!(matches!(
            issue_with(&mut store, KEY_A, &fixture()),
            Err(PublisherError::Store)
        ));
        assert!(matches!(
            issue_with(&mut store, KEY_A, &fixture()),
            Err(PublisherError::Store)
        ));
        drop(store);
        assert_eq!(DurableStore::open(&full).unwrap().count(), 0);
    }

    #[test]
    fn rename_unlink_hardlink_and_replacement_fail_stop() {
        for mode in ["rename", "unlink", "hardlink", "replace"] {
            let temp = secure_tempdir();
            let path = temp.path().join("publisher.ledger");
            let mut store = DurableStore::open(&path).unwrap();
            match mode {
                "rename" => std::fs::rename(&path, temp.path().join("moved")).unwrap(),
                "unlink" => std::fs::remove_file(&path).unwrap(),
                "hardlink" => hard_link(&path, temp.path().join("hard")).unwrap(),
                "replace" => {
                    std::fs::rename(&path, temp.path().join("old")).unwrap();
                    std::fs::write(&path, b"replacement").unwrap();
                    std::fs::set_permissions(&path, Permissions::from_mode(0o600)).unwrap();
                }
                _ => unreachable!(),
            }
            assert!(matches!(
                issue_with(&mut store, KEY_A, &fixture()),
                Err(PublisherError::Store)
            ));
        }
    }

    #[test]
    fn substitution_after_fsync_is_never_acknowledged() {
        let temp = secure_tempdir();
        let path = temp.path().join("publisher.ledger");
        let moved = temp.path().join("moved.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        let path_for_hook = path.clone();
        store.set_after_sync(move || std::fs::rename(path_for_hook, moved).unwrap());
        assert!(matches!(
            issue_with(&mut store, KEY_A, &fixture()),
            Err(PublisherError::Store)
        ));
        assert!(matches!(
            issue_with(&mut store, KEY_A, &fixture()),
            Err(PublisherError::Store)
        ));
    }

    #[test]
    fn parent_directory_rename_and_replacement_fail_stop() {
        let temp = secure_tempdir();
        let state = temp.path().join("state");
        std::fs::create_dir(&state).unwrap();
        std::fs::set_permissions(&state, Permissions::from_mode(0o700)).unwrap();
        let path = state.join("publisher.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        std::fs::rename(&state, temp.path().join("moved-state")).unwrap();
        std::fs::create_dir(&state).unwrap();
        std::fs::set_permissions(&state, Permissions::from_mode(0o700)).unwrap();
        assert!(matches!(
            issue_with(&mut store, KEY_A, &fixture()),
            Err(PublisherError::Store)
        ));
    }

    #[test]
    fn deadline_before_admission_rejects_and_admitted_commit_is_authoritative() {
        let temp = secure_tempdir();
        let path = temp.path().join("publisher.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        let fixture = fixture();
        let signer = TestSigner::new("test-publisher-v1");
        let identity = request_identity(&fixture.request_body);
        let request_sha = raw_request_sha256(&fixture.request_body);
        let auth = projection_digest(&fixture.request);
        let request_key = key_digest(KEY_A);
        let make_input = || IssueInput {
            request_key_sha256: &request_key,
            stable_authorization_sha256: &auth,
            request_identity: &identity,
            request_sha256: &request_sha,
            request_body: &fixture.request_body,
            request: &fixture.request,
            issued_at: 1_800_000_000,
            signature_domain: "test",
            signer: &signer,
        };
        assert!(
            store
                .issue_until(make_input(), Instant::now() - Duration::from_millis(1))
                .is_err()
        );
        assert_eq!(store.count(), 0);
        store.set_commit_delay(Duration::from_millis(600));
        let deadline = Instant::now() + Duration::from_millis(500);
        let response = store.issue_until(make_input(), deadline).unwrap();
        assert!(Instant::now() >= deadline);
        assert_eq!(issue_with(&mut store, KEY_A, &fixture).unwrap(), response);
    }

    #[test]
    fn exclusive_lock_allows_only_one_writer() {
        let temp = secure_tempdir();
        let path = temp.path().join("publisher.ledger");
        let store = DurableStore::open(&path).unwrap();
        assert!(DurableStore::open(&path).is_err());
        drop(store);
        assert!(DurableStore::open(&path).is_ok());
    }

    #[test]
    fn drop_unlocks_a_transiently_duplicated_open_file_description() {
        let temp = secure_tempdir();
        let path = temp.path().join("publisher.ledger");
        let store = DurableStore::open(&path).unwrap();
        let inherited = unsafe { libc::fcntl(store.file.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 3) };
        assert!(inherited >= 0);
        drop(store);
        let reopened = DurableStore::open(&path).unwrap();
        unsafe {
            libc::close(inherited);
        }
        drop(reopened);
    }

    #[test]
    fn concurrent_open_has_one_owner() {
        let temp = secure_tempdir();
        let path = Arc::new(temp.path().join("publisher.ledger"));
        let start = Arc::new(Barrier::new(8));
        let attempted = Arc::new(Barrier::new(8));
        let handles = (0..8)
            .map(|_| {
                let path = path.clone();
                let start = start.clone();
                let attempted = attempted.clone();
                thread::spawn(move || {
                    start.wait();
                    let store = DurableStore::open(&path);
                    let opened = store.is_ok();
                    attempted.wait();
                    drop(store);
                    opened
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .filter(|opened| *opened)
                .count(),
            1
        );
    }

    fn mutate_index_field(index: &mut EntryIndex, field: &str) {
        fn flip_first(value: &mut String) {
            let replacement = if value.starts_with('0') { "1" } else { "0" };
            value.replace_range(..1, replacement);
        }
        match field {
            "offset" => index.frame_offset += 1,
            "length" => index.frame_length -= 1,
            "sequence" => index.sequence += 1,
            "previous-hash" => index.previous_hash[0] ^= 1,
            "frame-hash" => index.frame_hash[0] ^= 1,
            "request-key" => flip_first(&mut index.request_key_sha256),
            "request-identity" => flip_first(&mut index.request_identity),
            "request-digest" => flip_first(&mut index.request_sha256),
            "stable-authorization" => flip_first(&mut index.stable_authorization_sha256),
            _ => unreachable!(),
        }
    }

    #[test]
    fn all_nine_index_substitutions_poison_before_exact_or_conflicting_return() {
        for first_attempt in ["exact", "conflict"] {
            for field in [
                "offset",
                "length",
                "sequence",
                "previous-hash",
                "frame-hash",
                "request-key",
                "request-identity",
                "request-digest",
                "stable-authorization",
            ] {
                let temp = secure_tempdir();
                let path = temp.path().join(format!("{first_attempt}-{field}.ledger"));
                let original = fixture();
                let conflicting = distinct_fixture();
                let mut store = DurableStore::open(&path).unwrap();
                let response = issue_with(&mut store, KEY_A, &original).unwrap();
                let durable = std::fs::read(&path).unwrap();
                let index = store.by_request_key.get_mut(&key_digest(KEY_A)).unwrap();
                mutate_index_field(index, field);

                let first = if first_attempt == "exact" {
                    issue_with(&mut store, KEY_A, &original)
                } else {
                    issue_with(&mut store, KEY_A, &conflicting)
                };
                assert!(
                    matches!(first, Err(PublisherError::Store)),
                    "{first_attempt}/{field}"
                );
                assert!(store.poisoned, "{first_attempt}/{field}");
                assert!(matches!(
                    issue_with(&mut store, KEY_A, &original),
                    Err(PublisherError::Store)
                ));
                assert!(matches!(
                    issue_with(&mut store, KEY_B, &conflicting),
                    Err(PublisherError::Store)
                ));
                assert_eq!(std::fs::read(&path).unwrap(), durable);
                drop(store);

                let mut restarted = DurableStore::open(&path).unwrap();
                assert_eq!(
                    issue_with(&mut restarted, KEY_A, &original).unwrap(),
                    response
                );
                issue_with(&mut restarted, KEY_B, &conflicting).unwrap();
                drop(restarted);
                assert_eq!(DurableStore::open(&path).unwrap().count(), 2);
            }
        }
    }

    #[test]
    fn hostile_whole_index_substitution_must_poison_map_binding() {
        for (slot_key, source_key, first_attempt) in
            [(KEY_A, KEY_B, "exact"), (KEY_B, KEY_A, "conflict")]
        {
            let temp = secure_tempdir();
            let path = temp
                .path()
                .join(format!("whole-index-{first_attempt}.ledger"));
            let first = fixture();
            let second = distinct_fixture();
            let third = third_fixture();
            let mut store = DurableStore::open(&path).unwrap();
            issue_with(&mut store, KEY_A, &first).unwrap();
            issue_with(&mut store, KEY_B, &second).unwrap();
            let durable = std::fs::read(&path).unwrap();
            let substituted = store
                .by_request_key
                .get(&key_digest(source_key))
                .unwrap()
                .clone();
            store
                .by_request_key
                .insert(key_digest(slot_key), substituted);

            let slot_fixture = if slot_key == KEY_A { &first } else { &second };
            let conflicting = if slot_key == KEY_A { &second } else { &first };
            let first_result = if first_attempt == "exact" {
                issue_with(&mut store, slot_key, slot_fixture)
            } else {
                issue_with(&mut store, slot_key, conflicting)
            };
            assert!(matches!(first_result, Err(PublisherError::Store)));
            assert!(
                store.poisoned,
                "valid-frame substitution was treated as caller conflict"
            );
            assert!(matches!(
                issue_with(&mut store, slot_key, slot_fixture),
                Err(PublisherError::Store)
            ));
            assert!(matches!(
                issue_with(&mut store, slot_key, conflicting),
                Err(PublisherError::Store)
            ));
            assert!(matches!(
                issue_with(&mut store, KEY_C, &third),
                Err(PublisherError::Store)
            ));
            assert!(store.poisoned);
            assert_eq!(std::fs::read(&path).unwrap(), durable);
            drop(store);

            let mut restarted = DurableStore::open(&path).unwrap();
            assert_eq!(restarted.count(), 2);
            issue_with(&mut restarted, KEY_A, &first).unwrap();
            issue_with(&mut restarted, KEY_B, &second).unwrap();
            issue_with(&mut restarted, KEY_C, &third).unwrap();
            drop(restarted);
            assert_eq!(DurableStore::open(&path).unwrap().count(), 3);
        }
    }

    fn worker_issue(key: &str, value: Fixture) -> crate::store_worker::StoreIssue {
        let request_identity = request_identity(&value.request_body);
        let request_sha256 = raw_request_sha256(&value.request_body);
        let stable_authorization_sha256 = projection_digest(&value.request);
        crate::store_worker::StoreIssue {
            request_key_sha256: key_digest(key),
            stable_authorization_sha256,
            request_identity,
            request_sha256,
            request_body: value.request_body,
            request: value.request,
            issued_at: 1_800_000_000,
            signature_domain: "test".into(),
            signer: Arc::new(TestSigner::new("test-publisher-v1")),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn queued_exact_conflicting_and_distinct_requests_cannot_ack_after_poison() {
        use std::sync::mpsc;

        let temp = secure_tempdir();
        let path = temp.path().join("worker-poison-race.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        let durable = std::fs::read(&path).unwrap();
        let (entered_tx, entered_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        store.set_before_indexed_decode(move |frame| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            frame[FRAME_PREFIX_BYTES] ^= 1;
        });
        let worker = Arc::new(crate::store_worker::StoreWorker::spawn(store, 128).unwrap());
        let exact_worker = worker.clone();
        let exact = tokio::spawn(async move {
            exact_worker
                .issue_until(
                    worker_issue(KEY_A, fixture()),
                    tokio::time::Instant::now() + Duration::from_secs(10),
                )
                .await
        });
        tokio::task::spawn_blocking(move || entered_rx.recv().unwrap())
            .await
            .unwrap();

        // Spawned requests need not all enter the queue before poisoning.
        let mut queued = Vec::new();
        for ordinal in 0..63 {
            let worker = worker.clone();
            queued.push(tokio::spawn(async move {
                let (key, value) = match ordinal % 3 {
                    0 => (KEY_A, fixture()),
                    1 => (KEY_A, distinct_fixture()),
                    _ => (KEY_B, distinct_fixture()),
                };
                worker
                    .issue_until(
                        worker_issue(key, value),
                        tokio::time::Instant::now() + Duration::from_secs(10),
                    )
                    .await
            }));
        }
        release_tx.send(()).unwrap();
        assert!(matches!(exact.await.unwrap(), Err(PublisherError::Store)));
        for result in queued {
            assert!(matches!(result.await.unwrap(), Err(PublisherError::Store)));
        }
        assert_eq!(std::fs::read(&path).unwrap(), durable);
        assert!(
            worker
                .shutdown_until(tokio::time::Instant::now() + Duration::from_secs(10))
                .await
        );
        drop(worker);
        assert_eq!(DurableStore::open(&path).unwrap().count(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn worker_unwind_disconnects_without_acknowledging() {
        let temp = secure_tempdir();
        let path = temp.path().join("worker-panic.ledger");
        let mut store = DurableStore::open(&path).unwrap();
        issue_with(&mut store, KEY_A, &fixture()).unwrap();
        let durable = std::fs::read(&path).unwrap();
        store.set_before_indexed_decode(|_| panic!("hostile lookup unwind"));
        let worker = crate::store_worker::StoreWorker::spawn(store, 8).unwrap();

        assert!(matches!(
            worker
                .issue_until(
                    worker_issue(KEY_A, fixture()),
                    tokio::time::Instant::now() + Duration::from_secs(10),
                )
                .await,
            Err(PublisherError::Store)
        ));
        assert!(matches!(
            worker
                .issue_until(
                    worker_issue(KEY_B, distinct_fixture()),
                    tokio::time::Instant::now() + Duration::from_secs(10),
                )
                .await,
            Err(PublisherError::Store)
        ));
        assert_eq!(std::fs::read(&path).unwrap(), durable);
        drop(worker);
        assert_eq!(DurableStore::open(&path).unwrap().count(), 1);
    }
}
