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

The compute-AQL queue records are bound by a third, independently named
KFD_AQL_QUEUE_LIFECYCLE_SCHEMA_MANIFEST under schema ID
linux-kfd-aql-queue-lifecycle-1.18-generic-ioc-v1, with SHA-256
b11f3c8c766dd25394350646e35269e10c8a33acb98f74cba2a82e95fa185c4e.
It includes the exact frozen R1 and R2 digests as prerequisites and adds the
queue ABI plus the exact reviewed gfx942 queue semantic source set. Neither
version admission nor the two prerequisite manifests authenticate this queue
schema.

Future VM or memory authority must bind the R1 and R2 manifests along with the
runtime device and process evidence. Future queue authority must additionally
bind the R4 queue manifest. R1 version or device admission alone does not
authorize ACQUIRE_VM, memory operations, or queue operations.

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
- `kfd_ioctl_create_queue_args`, `kfd_ioctl_update_queue_args`, and
  `kfd_ioctl_destroy_queue_args` with their three exact request numbers
- only compute-AQL queue construction (`KFD_IOC_QUEUE_TYPE_COMPUTE_AQL`), with
  unextended percentage 0 through 100, priority 0 through 15, and power-of-two
  ring sizes of at least 1024 bytes
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

The queue schema pins this exact reviewed gfx942 queue semantic source set:

- `amd/amdkfd/kfd_queue.c` SHA-256
  `fb4b2a5c9e6981222873bcd7aca7e9c1397cba8f1a6b33634d2a48d4427fe062`
- `amd/amdkfd/kfd_process_queue_manager.c` SHA-256
  `8526e258824dbe145e4209cf0fed26463729234ba24369f39e3413e7e6e028db`
- `amd/amdkfd/kfd_device_queue_manager.c` SHA-256
  `d61e53a78c1855c4badefbebb6c6ec52702be8cfe072253341c277337641c682`
- `amd/amdkfd/kfd_mqd_manager_v9.c` SHA-256
  `21166e9dbe2a4c24cbcd6f9ff6193aa093230e91fbafc8b4ac4eee1465cd2c9e`
- `amd/amdkfd/kfd_priv.h` SHA-256
  `f991330031c14725b2be0636ec1896ab530dc3d07d530ebd4f47efff97a82a99`
- `amd/amdkfd/kfd_device_queue_manager_v9.c` SHA-256
  `53021a6f8211212f872545403e200d34d2e8c49b1cbdd17e382ae7baa43e52f2`
- `amd/amdkfd/kfd_packet_manager.c` SHA-256
  `1ed642990cbb7d4cdbde211fee571318e233c19744ea1663d8eb68946c1310dd`
- `amd/amdkfd/kfd_kernel_queue.c` SHA-256
  `13e5d3634bcfed2ae871d8da0700cde47d8671eb014831b5d1ca95ed5a22fb36`
- `amd/amdkfd/kfd_mqd_manager.h` SHA-256
  `61ea7d4a13fb3168d0f026ecb13b13cf5846c86f233289043728b62ac9068605`
- `amd/amdkfd/kfd_device_queue_manager.h` SHA-256
  `9e43b8f41ad89d1dd21fddf38dff4182f09b01218778f8278a743eacb72ceadd`
- `amd/include/v9_structs.h` SHA-256
  `18f8e59e4cab35d579d2e3f9fc4eadffd81d518d586065de4d9d0ab4fcc131d7`
- `amd/amdgpu/amdgpu_amdkfd_gfx_v9.c` SHA-256
  `d112169b3231439086da4943c7675bb4aeddb111b483a687fdd95794710ab27c`
- `amd/amdgpu/amdgpu_amdkfd_gfx_v9.h` SHA-256
  `97bc6cd046c9c2495962d26d455e5231d95b0503385354177c366ea21fa9ed2e`
- `amd/include/asic_reg/gc/gc_9_0_offset.h` SHA-256
  `dde287260e0b63eecfd7b723c1fdfaf9a3da7155f0ccd331385b9acc09433aa5`
- `amd/include/asic_reg/gc/gc_9_0_sh_mask.h` SHA-256
  `f67f3f753231a53e82e39783313605cd382eb9727f2cda775d6e849a7c38063e`
- `amd/include/asic_reg/gc/gc_9_4_3_sh_mask.h` SHA-256
  `8ee3fb2c721703a1643c118502e2900bd622b4d8d287103bd53922f92d35611b`

This set covers the reviewed UAPI parsing, queue buffer acquisition,
per-process and device lifecycle, HWS packet path, gfx9 MQD programming,
KFD-to-KGD operations, MQD structures, and register definitions. It is not a
claim that these files form the complete transitive kernel build closure, and
their hashes do not authenticate the code loaded by the running kernel.

## Compute-AQL queue boundary

The safe queue-create constructor fixes `queue_type` to compute AQL,
`sdma_engine_id` and `pad` to zero, and initializes the kernel-written queue ID
and doorbell offset to fail-closed sentinels. Typed admission rejects rings the
driver would clamp, non-power-of-two rings, target-XCC bits repurposed through
`queue_percentage`, percentages above 100, and priorities above 15. The
reconfiguration constructor additionally requires a typed nonzero numeric ring
address observation. That observation makes no provenance, mapping, alignment,
or ownership claim.

The separate disable constructor deliberately emits a zero ring address, the
driver's reviewed disable signal. It still requires an admitted nonzero ring
size because the driver stores the size and updates the MQD, fixes percentage
to zero, and admits priority normally. A failed UPDATE does not establish
whether the queue is active or disabled. The destroy constructor zeros its
reserved padding.

The remaining create fields are represented exactly but are deliberately raw.
Ring, read-pointer, write-pointer, EOP, and CWSR addresses are opaque integers;
EOP, CWSR, and control-stack sizes depend on the selected GPU topology. This
crate cannot prove that any address names a live allocation, that the allocation
is large enough and mapped to the selected GPU, or that device-derived auxiliary
sizes are current. Those checks belong to a lifetime-bound queue adapter that
composes R1 device evidence, R2 allocation evidence, and the R4 schema.

The returned queue ID is not authority and the returned doorbell offset is not
a mapping. UPDATE carries no queue-format field, so its builder name does not
prove that the numeric ID belongs to a live AQL queue. DESTROY likewise does not
prove ownership or one-shot lifecycle state. No queue request is exposed as a
method on `AdmittedKfdUapi`, because a successful KFD 1.18 version query does
not authenticate the queue schema or any active driver semantics.

The separate gfx942 queue-resource output schema admits only numeric outputs
from an already successful CREATE_QUEUE operation. It records that process
queue slot zero is valid, bounds the active PQM slot to 0..1023, and validates
the non-MES doorbell mmap type, GPU-ID hash, 8192-byte process slice, and
8-byte offset alignment. Its types still grant no queue, mmap, MMIO, or
doorbell-store authority. The caller must independently validate ioctl success,
unchanged inputs, allocation ownership, currentness, and lifecycle state.
The output decomposition clears the full 8191-byte slice mask to produce a
canonical encoded process-slice offset observation and retains the low bits as
the in-process byte offset. This follows active ROCr's SOC15 path. Clearing only
the 4095-byte page mask would incorrectly retain the second-page selector for
offsets from 4096 through 8184. Active KFD requires the doorbell VMA length to
equal the complete 8192-byte non-MES process slice; it selects the device from
the encoded high fields and maps that whole allocation. The decomposed
integers remain observations, not an executable mmap plan.

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

Compile-time assertions, `tests/kfd_uapi_1_18.rs`, and
`tests/kfd_aql_queue_uapi_1_18.rs` pin every admitted struct size, alignment,
field offset, request number, and typed queue range to independent golden
values and hostile boundary cases.
`KFD_UAPI_SCHEMA_MANIFEST` canonically binds only the frozen R1 facts.
`KFD_MEMORY_LIFECYCLE_SCHEMA_MANIFEST` separately binds the compositional R2
memory facts and the exact R1 digest it requires. The R4 queue manifest binds
both prerequisite digests independently. All three SHA-256 values are
recomputed in tests. These manifests identify reviewed userspace content;
running kernel, module, boot, device, and process identities remain separate
contracted observations.

The independent C ABI oracle is preserved at
`tests/oracles/kfd_uapi_1_18.c`. On the reviewed host it is built directly
against the active header with:

```text
cc -std=c11 -Wall -Wextra -Werror \
  -I/usr/src/amdgpu-6.16.13-2341068.24.04/include/uapi \
  tests/oracles/kfd_uapi_1_18.c -o /tmp/kfd-uapi-oracle
```

`tests/oracles/run-kfd-gfx942-queue-resources-oracle.sh` separately hashes
the exact active kernel and ROCr source set, compiles a C formula oracle
against the active KFD header, and prints the queue-ID, CWSR, ring, EOP,
counter, doorbell, and ROCr flag goldens. This source-profile oracle is
read-only and creates no queue.

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
rollback, executable loading, queue syscalls, queue ownership and rollback,
doorbell mmap and stores, AQL packet encoding, signal handling, SVM/VRAM/peer
allocation and mapping, and all syscall execution remain outside this crate.
SDMA, PM4 compute, XGMI, target-XCC selection, CU masks, GWS, queue priority
policy, queue preemption, CWSR allocation, EOP allocation, persistent-queue
policy, and multi-process queue sharing are not admitted by this initial queue
profile. The reset constants describe a prospective whole-GPU event stream,
not an all-reset generation. In particular, this crate is not a safe wrapper
around `/dev/kfd`; it is the bounded data-only input to that wrapper.
