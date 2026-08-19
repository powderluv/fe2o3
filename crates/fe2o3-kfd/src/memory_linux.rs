//! Minimal unsafe Linux boundary for the owned memory transaction.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::os::fd::AsRawFd;

use fe2o3_kfd_uapi::{
    AMDKFD_IOC_ACQUIRE_VM, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, AMDKFD_IOC_FREE_MEMORY_OF_GPU,
    AMDKFD_IOC_MAP_MEMORY_TO_GPU, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, KfdAllocMemoryFlags,
    KfdIoctlAcquireVmArgs, KfdIoctlAllocMemoryOfGpuArgs, KfdIoctlFreeMemoryOfGpuArgs,
    KfdIoctlMapMemoryToGpuArgs, KfdIoctlUnmapMemoryFromGpuArgs,
};
use fe2o3_runtime_model::{ModelDeviceAdmissionV1, ModelVmAdmissionV1, VmIdV1};
use rustix::ioctl::{Opcode, Setter, Updater};
use rustix::mm::{Advice, MapFlags, MprotectFlags, ProtFlags};

use super::memory::{KernelOutcome, MemoryBackend, MemorySessionError};
use crate::{CheckedGfx942XnackMinusDevice, InclusiveAperture};

const ACQUIRE_VM_OPCODE: Opcode = AMDKFD_IOC_ACQUIRE_VM as Opcode;
const ALLOC_MEMORY_OPCODE: Opcode = AMDKFD_IOC_ALLOC_MEMORY_OF_GPU as Opcode;
const FREE_MEMORY_OPCODE: Opcode = AMDKFD_IOC_FREE_MEMORY_OF_GPU as Opcode;
const MAP_MEMORY_OPCODE: Opcode = AMDKFD_IOC_MAP_MEMORY_TO_GPU as Opcode;
const UNMAP_MEMORY_OPCODE: Opcode = AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU as Opcode;
#[cfg(feature = "live-validation")]
const LINUX_ENOMEM: i32 = 12;

#[cfg(feature = "live-validation")]
unsafe extern "C" {
    fn mincore(address: *mut c_void, length: usize, residency: *mut u8) -> i32;
}

pub(super) struct LinuxMemoryBackend {
    device: CheckedGfx942XnackMinusDevice,
}

pub(super) struct LinuxVaReservation {
    address: NonNull<c_void>,
    bytes: usize,
    replaced: bool,
}

pub(super) struct LinuxCpuMapping {
    address: NonNull<c_void>,
    bytes: usize,
    active: bool,
    accessible: bool,
}

impl LinuxMemoryBackend {
    pub(super) fn new(device: CheckedGfx942XnackMinusDevice) -> Self {
        Self { device }
    }

    pub(super) fn bind_model_vm(
        &mut self,
        vm_id: VmIdV1,
    ) -> Result<ModelVmAdmissionV1, MemorySessionError> {
        self.device
            .register_memory_vm_model_only(vm_id)
            .map_err(MemorySessionError::Device)
    }

    pub(super) fn model_device(&self) -> ModelDeviceAdmissionV1 {
        self.device.model_admission()
    }

    pub(super) fn model_aperture(&self) -> InclusiveAperture {
        self.device.observation().aperture().gpuvm()
    }

    fn discard_unprepared_mapping_or_abort(mapping: &mut LinuxCpuMapping) {
        if !mapping.active {
            return;
        }
        // SAFETY: no readable/writable access has been enabled and no slice has
        // been formed. Returning an ambiguously inheritable VMA would violate
        // the safe API contract, so failed synchronous cleanup is fail-stop.
        if unsafe { rustix::mm::munmap(mapping.address.as_ptr(), mapping.bytes) }.is_err() {
            std::process::abort();
        }
        mapping.active = false;
        mapping.accessible = false;
    }

    #[cfg(feature = "live-validation")]
    pub(super) fn verify_dontfork_child_negative(
        &self,
        mapping: &LinuxCpuMapping,
    ) -> Result<(), MemorySessionError> {
        let tasks = std::fs::read_dir("/proc/self/task")
            .map_err(|_| MemorySessionError::ChildProbe("read /proc/self/task"))?
            .take(2)
            .count();
        if tasks != 1 {
            return Err(MemorySessionError::IsolationRequired);
        }
        let mut residency = vec![
            0_u8;
            mapping.bytes.div_ceil(
                super::memory::HOST_VISIBLE_MEMORY_PAGE_BYTES_V1 as usize
            )
        ];
        // SAFETY: this probe admits fork only after observing exactly one task,
        // holds no user-visible mapping borrow, and performs only mincore then
        // exit_group in the child. The parent synchronously waits.
        match unsafe { rustix::runtime::kernel_fork() }
            .map_err(|source| Self::syscall("fork DONTFORK child probe", source))?
        {
            rustix::runtime::Fork::Child(_) => {
                // SAFETY: mincore treats the address as an integer range and
                // reports ENOMEM for a DONTFORK-removed VMA; it does not
                // dereference the absent userspace mapping.
                let result = unsafe {
                    mincore(
                        mapping.address.as_ptr(),
                        mapping.bytes,
                        residency.as_mut_ptr(),
                    )
                };
                let code = if result == -1
                    && std::io::Error::last_os_error().raw_os_error() == Some(LINUX_ENOMEM)
                {
                    0
                } else if result == 0 {
                    1
                } else {
                    2
                };
                rustix::runtime::exit_group(code);
            }
            rustix::runtime::Fork::ParentOf(child) => {
                let (_, status) =
                    rustix::process::waitpid(Some(child), rustix::process::WaitOptions::empty())
                        .map_err(|source| Self::syscall("wait DONTFORK child probe", source))?
                        .ok_or(MemorySessionError::ChildProbe("child was not waitable"))?;
                match status.exit_status() {
                    Some(0) => Ok(()),
                    Some(1) => Err(MemorySessionError::DontForkMappingInherited),
                    _ => Err(MemorySessionError::ChildProbe("child mincore protocol")),
                }
            }
        }
    }

    fn syscall(operation: &'static str, source: rustix::io::Errno) -> MemorySessionError {
        MemorySessionError::Syscall { operation, source }
    }

    fn exact_progress(
        operation: &'static str,
        handle: u64,
        gpu_id: u32,
        old_success: u32,
        unmap: bool,
        kfd: &rustix::fd::OwnedFd,
    ) -> KernelOutcome<u32> {
        let device_ids = [gpu_id];
        let pointer = device_ids.as_ptr() as usize as u64;
        if unmap {
            let mut args = KfdIoctlUnmapMemoryFromGpuArgs::retry(
                handle,
                pointer,
                device_ids.len() as u32,
                old_success,
            );
            // SAFETY: the opcode and LP64 layout are frozen by the KFD 1.18
            // oracle. `device_ids` and the initialized in/out record remain
            // live and immutably located for the complete call.
            let request = unsafe { Updater::<UNMAP_MEMORY_OPCODE, _>::new(&mut args) };
            // SAFETY: request, nested pointer, lengths, and exclusive output
            // borrow are established above. The result remains untrusted.
            let result = unsafe { rustix::ioctl::ioctl(kfd, request) }
                .map_err(|source| Self::syscall(operation, source));
            if args.handle != handle
                || args.device_ids_array_ptr != pointer
                || args.n_devices != 1
                || args.n_success < old_success
                || args.n_success > 1
            {
                return KernelOutcome {
                    value: args.n_success,
                    result: Err(MemorySessionError::KernelResultMalformed(
                        "UNMAP_MEMORY_FROM_GPU immutable request or cumulative progress",
                    )),
                };
            }
            KernelOutcome {
                value: args.n_success,
                result,
            }
        } else {
            let mut args = KfdIoctlMapMemoryToGpuArgs::retry(
                handle,
                pointer,
                device_ids.len() as u32,
                old_success,
            );
            // SAFETY: same reviewed nested-pointer contract as the unmap path.
            let request = unsafe { Updater::<MAP_MEMORY_OPCODE, _>::new(&mut args) };
            // SAFETY: request and backing array stay live; output is exclusive.
            let result = unsafe { rustix::ioctl::ioctl(kfd, request) }
                .map_err(|source| Self::syscall(operation, source));
            if args.handle != handle
                || args.device_ids_array_ptr != pointer
                || args.n_devices != 1
                || args.n_success < old_success
                || args.n_success > 1
            {
                return KernelOutcome {
                    value: args.n_success,
                    result: Err(MemorySessionError::KernelResultMalformed(
                        "MAP_MEMORY_TO_GPU immutable request or cumulative progress",
                    )),
                };
            }
            KernelOutcome {
                value: args.n_success,
                result,
            }
        }
    }
}

impl MemoryBackend for LinuxMemoryBackend {
    type Reservation = LinuxVaReservation;
    type Mapping = LinuxCpuMapping;

    fn opener_pid(&self) -> u32 {
        self.device.process_incarnation().pid()
    }

    fn gpu_id(&self) -> u32 {
        self.device.observation().kfd_gpu_id()
    }

    fn gpuvm_aperture(&self) -> InclusiveAperture {
        self.device.observation().aperture().gpuvm()
    }

    fn page_size(&self) -> usize {
        rustix::param::page_size()
    }

    fn check_currentness(&mut self) -> Result<(), MemorySessionError> {
        self.device.check_observable_currentness()?;
        Ok(())
    }

    fn acquire_vm(&mut self) -> Result<(), MemorySessionError> {
        let raw_fd = self.device.render_fd.as_raw_fd();
        let drm_fd = u32::try_from(raw_fd)
            .map_err(|_| MemorySessionError::KernelResultMalformed("render descriptor number"))?;
        let args = KfdIoctlAcquireVmArgs::new(drm_fd, self.gpu_id());
        // SAFETY: opcode and input-only C layout are frozen by the independent
        // KFD 1.18 oracle. Both retained descriptors outlive the call.
        let request = unsafe { Setter::<ACQUIRE_VM_OPCODE, _>::new(args) };
        // SAFETY: the input-only request and retained KFD descriptor satisfy
        // the reviewed request contract. Success is rechecked for currentness.
        unsafe { rustix::ioctl::ioctl(&self.device.kfd.opened.fd, request) }
            .map_err(|source| Self::syscall("AMDKFD_IOC_ACQUIRE_VM", source))?;
        self.device.retire_model_on_drop = false;
        Ok(())
    }

    fn reserve_va(&mut self, bytes: usize) -> Result<Self::Reservation, MemorySessionError> {
        // SAFETY: null lets the kernel select a fresh range; a nonzero,
        // page-rounded length is supplied. No references exist to the result.
        let address = unsafe {
            rustix::mm::mmap_anonymous(
                core::ptr::null_mut(),
                bytes,
                ProtFlags::empty(),
                MapFlags::PRIVATE | MapFlags::NORESERVE,
            )
        }
        .map_err(|source| Self::syscall("reserve anonymous GPU VA", source))?;
        let address = NonNull::new(address).ok_or(MemorySessionError::KernelResultMalformed(
            "anonymous mmap address",
        ))?;
        Ok(LinuxVaReservation {
            address,
            bytes,
            replaced: false,
        })
    }

    fn reservation_address(reservation: &Self::Reservation) -> u64 {
        reservation.address.as_ptr() as usize as u64
    }

    fn alloc(&mut self, va: u64, bytes: u64) -> KernelOutcome<KfdIoctlAllocMemoryOfGpuArgs> {
        let mut args = KfdIoctlAllocMemoryOfGpuArgs::new(
            va,
            bytes,
            self.gpu_id(),
            KfdAllocMemoryFlags::HOST_VISIBLE_COHERENT,
        );
        // SAFETY: the opcode and in/out C layout are frozen by the KFD 1.18
        // oracle, and initialized exclusive storage remains live for the call.
        let request = unsafe { Updater::<ALLOC_MEMORY_OPCODE, _>::new(&mut args) };
        // SAFETY: request contract is established above; every field is still
        // treated as untrusted even if ioctl returns success.
        let result = unsafe { rustix::ioctl::ioctl(&self.device.kfd.opened.fd, request) }
            .map_err(|source| Self::syscall("AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", source));
        KernelOutcome {
            value: args,
            result,
        }
    }

    fn map_cpu(
        &mut self,
        reservation: &mut Self::Reservation,
        mmap_offset: u64,
        bytes: usize,
    ) -> Result<Self::Mapping, MemorySessionError> {
        if reservation.replaced || reservation.bytes != bytes {
            return Err(MemorySessionError::KernelResultMalformed(
                "VA reservation replacement",
            ));
        }
        // SAFETY: this exact anonymous reservation is owned and has no Rust
        // references. GPU VA and CPU VMA remain distinct authorities.
        unsafe { rustix::mm::munmap(reservation.address.as_ptr(), reservation.bytes) }
            .map_err(|source| Self::syscall("release anonymous GPU VA reservation", source))?;
        reservation.replaced = true;
        // SAFETY: null requests a kernel-selected CPU VMA. It is deliberately
        // PROT_NONE until DONTFORK succeeds, so the setup gap cannot expose BO
        // bytes even if an external raw fork violates the named contract.
        let mapped = unsafe {
            rustix::mm::mmap(
                core::ptr::null_mut(),
                bytes,
                ProtFlags::empty(),
                MapFlags::SHARED,
                &self.device.render_fd,
                mmap_offset,
            )
        }
        .map_err(|source| Self::syscall("mmap AMDGPU BO", source))?;
        let Some(address) = NonNull::new(mapped) else {
            // A mapping at address zero is outside the admitted profile. It is
            // still a live VMA and must not be returned ambiguously.
            // SAFETY: `mapped` and `bytes` are the exact successful mmap range.
            if unsafe { rustix::mm::munmap(mapped, bytes) }.is_err() {
                std::process::abort();
            }
            return Err(MemorySessionError::KernelResultMalformed(
                "AMDGPU BO mmap address",
            ));
        };
        Ok(LinuxCpuMapping {
            address,
            bytes,
            active: true,
            accessible: false,
        })
    }

    fn prepare_cpu_mapping(
        &mut self,
        mapping: &mut Self::Mapping,
    ) -> Result<(), MemorySessionError> {
        if !mapping.active || mapping.accessible {
            return Err(MemorySessionError::KernelResultMalformed(
                "CPU mapping setup state",
            ));
        }
        // SAFETY: the mapping is live, page-aligned, exclusively borrowed, and
        // PROT_NONE. DONTFORK is mandatory because TTM lacks VM_DONTCOPY and
        // would otherwise create a child VMA/BO reference.
        let advised = unsafe {
            rustix::mm::madvise(
                mapping.address.as_ptr(),
                mapping.bytes,
                Advice::LinuxDontFork,
            )
        };
        if let Err(source) = advised {
            Self::discard_unprepared_mapping_or_abort(mapping);
            return Err(Self::syscall("madvise MADV_DONTFORK", source));
        }
        // SAFETY: the exact still-live VMA has DONTFORK installed and no slice
        // exists. Read/write access is enabled only after that ordering point.
        let protected = unsafe {
            rustix::mm::mprotect(
                mapping.address.as_ptr(),
                mapping.bytes,
                MprotectFlags::READ | MprotectFlags::WRITE,
            )
        };
        if let Err(source) = protected {
            Self::discard_unprepared_mapping_or_abort(mapping);
            return Err(Self::syscall("mprotect AMDGPU BO read/write", source));
        }
        mapping.accessible = true;
        Ok(())
    }

    fn map_gpu(&mut self, handle: u64, old_success: u32) -> KernelOutcome<u32> {
        Self::exact_progress(
            "AMDKFD_IOC_MAP_MEMORY_TO_GPU",
            handle,
            self.gpu_id(),
            old_success,
            false,
            &self.device.kfd.opened.fd,
        )
    }

    fn unmap_gpu(&mut self, handle: u64, old_success: u32) -> KernelOutcome<u32> {
        Self::exact_progress(
            "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU",
            handle,
            self.gpu_id(),
            old_success,
            true,
            &self.device.kfd.opened.fd,
        )
    }

    fn with_bytes<R>(
        mapping: &Self::Mapping,
        requested_bytes: usize,
        f: impl FnOnce(&[u8]) -> R,
    ) -> R {
        debug_assert!(mapping.active && mapping.accessible && requested_bytes <= mapping.bytes);
        // SAFETY: the live mapping covers this range. The safe engine checks
        // phase and process before entering this boundary.
        let bytes = unsafe {
            core::slice::from_raw_parts(mapping.address.as_ptr().cast(), requested_bytes)
        };
        f(bytes)
    }

    fn with_bytes_mut<R>(
        mapping: &mut Self::Mapping,
        requested_bytes: usize,
        f: impl FnOnce(&mut [u8]) -> R,
    ) -> R {
        debug_assert!(mapping.active && mapping.accessible && requested_bytes <= mapping.bytes);
        // SAFETY: the exclusive mapping borrow covers the slice, and the safe
        // engine checks phase and process before entering this boundary.
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(mapping.address.as_ptr().cast(), requested_bytes)
        };
        f(bytes)
    }

    fn unmap_cpu(&mut self, mapping: &mut Self::Mapping) -> Result<(), MemorySessionError> {
        if !mapping.active || !mapping.accessible {
            return Err(MemorySessionError::KernelResultMalformed(
                "CPU mapping state",
            ));
        }
        // SAFETY: the mapping is exclusively borrowed and no safe slice can
        // survive a closure call. Explicit munmap must precede FREE.
        unsafe { rustix::mm::munmap(mapping.address.as_ptr(), mapping.bytes) }
            .map_err(|source| Self::syscall("munmap AMDGPU BO", source))?;
        mapping.active = false;
        mapping.accessible = false;
        Ok(())
    }

    fn free(&mut self, handle: u64) -> Result<(), MemorySessionError> {
        let args = KfdIoctlFreeMemoryOfGpuArgs::new(handle);
        // SAFETY: the input-only opcode/layout are oracle-frozen. The safe
        // engine invokes this operation at most once.
        let request = unsafe { Setter::<FREE_MEMORY_OPCODE, _>::new(args) };
        // SAFETY: request and retained KFD descriptor satisfy that contract.
        unsafe { rustix::ioctl::ioctl(&self.device.kfd.opened.fd, request) }
            .map_err(|source| Self::syscall("AMDKFD_IOC_FREE_MEMORY_OF_GPU", source))
    }
}

impl Drop for LinuxVaReservation {
    fn drop(&mut self) {
        // Deliberately no implicit munmap after an ambiguous operation.
    }
}

impl Drop for LinuxCpuMapping {
    fn drop(&mut self) {
        // Deliberately no implicit munmap or FREE retry.
    }
}
