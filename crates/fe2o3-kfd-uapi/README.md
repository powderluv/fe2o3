# fe2o3-kfd-uapi

Reviewed, `no_std`-compatible raw definitions for the first fe2o3 direct-KFD
runtime slice. This crate deliberately does not open devices, discover topology,
issue syscalls, or own resources.

## Reviewed schemas

`linux-kfd-uapi-1.18-generic-ioc-v1` was transcribed from the active AMDGPU DKMS
driver source installed on the MI300X development host:

- `amdgpu-dkms` package `1:6.16.13.30300400-2341068.24.04`
- `/usr/src/amdgpu-6.16.13-2341068.24.04/include/uapi/linux/kfd_ioctl.h`
  SHA-256 `b3721c1a428a32bb9994af579432af48c44fa65abb860049f11a63a5c093235d`
- `/usr/include/asm-generic/ioctl.h` SHA-256
  `76396e5537d75285c3ca20e3b6a79b101eebfdc14d39c104ff7eab778672160e`
- header-declared and running `/dev/kfd` UAPI version `1.18`

The frozen R1 discovery and identity manifest is
KFD_UAPI_SCHEMA_MANIFEST, with SHA-256
e4aad5d8e3177ea6d70298adab7741c377cb091373553ce689f3525e7514d9b4.
It remains byte-for-byte stable so an R2 data-definition extension cannot
silently broaden existing R1 admission evidence.

The memory-lifecycle definitions are bound independently by
KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST under schema ID
linux-kfd-memory-lifecycle-1.18-generic-ioc-v1, with SHA-256
e2d6987b7c8e61a405b2f775d5d004f458a096241459e4cfdf90bd4497f4d58a.
That manifest composes with R1 by including its exact schema ID and digest,
then binds the active KFD header and GPUVM implementation provenance, target,
package, allocation flags and profiles, layouts, and request numbers.

Future VM or memory authority must bind both manifests along with the runtime
device and process evidence. R1 version or device admission alone does not
authorize ACQUIRE_VM or any allocation, mapping, or free request.

The committed slice contains only:

- `kfd_ioctl_get_version_args` and `AMDKFD_IOC_GET_VERSION`
- `kfd_process_device_apertures`,
  `kfd_ioctl_get_process_apertures_new_args`, and
  `AMDKFD_IOC_GET_PROCESS_APERTURES_NEW`
- `kfd_ioctl_acquire_vm_args` and `AMDKFD_IOC_ACQUIRE_VM`
- `kfd_ioctl_alloc_memory_of_gpu_args`,
  `kfd_ioctl_free_memory_of_gpu_args`,
  `kfd_ioctl_map_memory_to_gpu_args`, and
  `kfd_ioctl_unmap_memory_from_gpu_args` with their four request numbers
- only the GTT, writable, executable, AQL-queue, coherent, and uncached raw
  allocation bits, exposed through four exact admitted profiles: ordinary
  host-visible coherent, kernarg, AQL queue, and host-visible executable
- `kfd_ioctl_set_xnack_mode_args` and `AMDKFD_IOC_SET_XNACK_MODE`
- `kfd_ioctl_smi_events_args`, `AMDKFD_IOC_SMI_EVENTS`, and only the whole-GPU
  pre/post-reset event indices and mask
- the generic Linux `_IOC` encoding needed by those requests
- exact-version admission evidence

Reviewed event and memory behavior is pinned to these active implementation
sources:

- `amd/amdkfd/kfd_smi_events.c` SHA-256
  `2d786562fe1e97b8257841b755106c8bce47658a2aa3b439ce4e0178323004bd`
- `amd/amdkfd/kfd_device.c` SHA-256
  `ccf20227c5cdd5b258758f50f61bbc1008a09ea776c101f035f83963e7d23037`
- `amd/amdkfd/kfd_chardev.c` SHA-256
  `f9a8805c5d479faee25e457051aa428e4bb523ecf1c7b1618a6a5f79ca5d7bba`
- `amd/amdgpu/amdgpu_amdkfd_gpuvm.c` SHA-256
  `c7cca2ee47a08c99bb73906662d82dd7d0b5738468fbef54848e5e6dd62ba50d`

The memory records are wire data, not resource wrappers. Addresses and handles
remain opaque integers. Allocation construction zeros the output handle and
mmap offset. Map and unmap construction distinguishes an initial request from a
retry, preserving the kernel-written `n_success` prefix for recovery after a
partial failure. Buffer provenance, lifetime, bounds, mmap, ownership,
rollback, and syscall execution belong to the later adapter.

Allocation admission is exact-match, not a permissive bit mask. VRAM,
USERPTR/SVM, doorbells, MMIO remaps, public allocation, extended coherency,
contiguous allocation, and every unknown bit pattern are rejected by the typed
admission path. The data-only map records can describe the kernel ABI's device
array, but they do not authorize peer mapping; the future adapter must correlate
every array element to the allocation's admitted device.

Compile-time assertions and `tests/kfd_uapi_1_18.rs` pin every struct size,
alignment, field offset, and request number to independent golden values.
`KFD_UAPI_SCHEMA_MANIFEST` canonically binds only the frozen R1 facts.
`KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST` separately binds the compositional R2
memory facts and the exact R1 digest it requires. Both SHA-256 values are
recomputed in tests. These manifests identify reviewed userspace content;
running kernel, module, boot, device, and process identities remain separate
contracted observations.

The independent C oracle is preserved at
`tests/oracles/kfd_uapi_1_18.c`. On the reviewed host it is built directly
against the active header with:

```text
cc -std=c11 -Wall -Wextra -Werror \
  -I/usr/src/amdgpu-6.16.13-2341068.24.04/include/uapi \
  tests/oracles/kfd_uapi_1_18.c -o /tmp/kfd-uapi-oracle
```

## Fail-closed boundary

The initial schema accepts exactly UAPI `1.18`. Linux minor UAPI revisions are
normally backwards compatible, but accepting an unreviewed revision would make
the crate's assurance claim broader than its evidence. Supporting another minor
version requires a named schema update, header-oracle comparison, and reviewed
compatibility tests.

The request encoder models Linux's generic `_IOC` bit layout used by the x86_64
MI300X runtime target. An architecture that overrides that layout requires a
separate reviewed schema.

## Not yet supported

VM ownership, virtual-address reservation, CPU mmap, memory ownership and
rollback, executable loading, queue creation, general event/signal handling,
SVM/VRAM/peer allocation and mapping, and syscall execution remain outside this
crate. The reset constants describe a prospective whole-GPU event stream, not
an all-reset generation. In particular, this crate is not a safe wrapper around
`/dev/kfd`; it is the bounded data-only input to that wrapper.
