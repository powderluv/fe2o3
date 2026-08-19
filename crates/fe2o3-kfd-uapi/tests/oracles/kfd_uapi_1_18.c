#include <linux/kfd_ioctl.h>
#include <stddef.h>
#include <stdio.h>

int main(void) {
    printf("get_version:size=%zu align=%zu major=%zu minor=%zu request=%#lx\n",
           sizeof(struct kfd_ioctl_get_version_args),
           _Alignof(struct kfd_ioctl_get_version_args),
           offsetof(struct kfd_ioctl_get_version_args, major_version),
           offsetof(struct kfd_ioctl_get_version_args, minor_version),
           (unsigned long)AMDKFD_IOC_GET_VERSION);

    printf("process_apertures:size=%zu align=%zu lds_base=%zu lds_limit=%zu "
           "scratch_base=%zu scratch_limit=%zu gpuvm_base=%zu gpuvm_limit=%zu "
           "gpu_id=%zu pad=%zu\n",
           sizeof(struct kfd_process_device_apertures),
           _Alignof(struct kfd_process_device_apertures),
           offsetof(struct kfd_process_device_apertures, lds_base),
           offsetof(struct kfd_process_device_apertures, lds_limit),
           offsetof(struct kfd_process_device_apertures, scratch_base),
           offsetof(struct kfd_process_device_apertures, scratch_limit),
           offsetof(struct kfd_process_device_apertures, gpuvm_base),
           offsetof(struct kfd_process_device_apertures, gpuvm_limit),
           offsetof(struct kfd_process_device_apertures, gpu_id),
           offsetof(struct kfd_process_device_apertures, pad));

    printf("get_process_apertures_new:size=%zu align=%zu pointer=%zu nodes=%zu "
           "pad=%zu request=%#lx\n",
           sizeof(struct kfd_ioctl_get_process_apertures_new_args),
           _Alignof(struct kfd_ioctl_get_process_apertures_new_args),
           offsetof(struct kfd_ioctl_get_process_apertures_new_args,
                    kfd_process_device_apertures_ptr),
           offsetof(struct kfd_ioctl_get_process_apertures_new_args,
                    num_of_nodes),
           offsetof(struct kfd_ioctl_get_process_apertures_new_args, pad),
           (unsigned long)AMDKFD_IOC_GET_PROCESS_APERTURES_NEW);

    printf("acquire_vm:size=%zu align=%zu drm_fd=%zu gpu_id=%zu request=%#lx\n",
           sizeof(struct kfd_ioctl_acquire_vm_args),
           _Alignof(struct kfd_ioctl_acquire_vm_args),
           offsetof(struct kfd_ioctl_acquire_vm_args, drm_fd),
           offsetof(struct kfd_ioctl_acquire_vm_args, gpu_id),
           (unsigned long)AMDKFD_IOC_ACQUIRE_VM);

    printf("alloc_memory:size=%zu align=%zu va_addr=%zu size_field=%zu "
           "handle=%zu mmap_offset=%zu gpu_id=%zu flags=%zu request=%#lx\n",
           sizeof(struct kfd_ioctl_alloc_memory_of_gpu_args),
           _Alignof(struct kfd_ioctl_alloc_memory_of_gpu_args),
           offsetof(struct kfd_ioctl_alloc_memory_of_gpu_args, va_addr),
           offsetof(struct kfd_ioctl_alloc_memory_of_gpu_args, size),
           offsetof(struct kfd_ioctl_alloc_memory_of_gpu_args, handle),
           offsetof(struct kfd_ioctl_alloc_memory_of_gpu_args, mmap_offset),
           offsetof(struct kfd_ioctl_alloc_memory_of_gpu_args, gpu_id),
           offsetof(struct kfd_ioctl_alloc_memory_of_gpu_args, flags),
           (unsigned long)AMDKFD_IOC_ALLOC_MEMORY_OF_GPU);

    printf("free_memory:size=%zu align=%zu handle=%zu request=%#lx\n",
           sizeof(struct kfd_ioctl_free_memory_of_gpu_args),
           _Alignof(struct kfd_ioctl_free_memory_of_gpu_args),
           offsetof(struct kfd_ioctl_free_memory_of_gpu_args, handle),
           (unsigned long)AMDKFD_IOC_FREE_MEMORY_OF_GPU);

    printf("map_memory:size=%zu align=%zu handle=%zu pointer=%zu devices=%zu "
           "success=%zu request=%#lx\n",
           sizeof(struct kfd_ioctl_map_memory_to_gpu_args),
           _Alignof(struct kfd_ioctl_map_memory_to_gpu_args),
           offsetof(struct kfd_ioctl_map_memory_to_gpu_args, handle),
           offsetof(struct kfd_ioctl_map_memory_to_gpu_args, device_ids_array_ptr),
           offsetof(struct kfd_ioctl_map_memory_to_gpu_args, n_devices),
           offsetof(struct kfd_ioctl_map_memory_to_gpu_args, n_success),
           (unsigned long)AMDKFD_IOC_MAP_MEMORY_TO_GPU);

    printf("unmap_memory:size=%zu align=%zu handle=%zu pointer=%zu devices=%zu "
           "success=%zu request=%#lx\n",
           sizeof(struct kfd_ioctl_unmap_memory_from_gpu_args),
           _Alignof(struct kfd_ioctl_unmap_memory_from_gpu_args),
           offsetof(struct kfd_ioctl_unmap_memory_from_gpu_args, handle),
           offsetof(struct kfd_ioctl_unmap_memory_from_gpu_args, device_ids_array_ptr),
           offsetof(struct kfd_ioctl_unmap_memory_from_gpu_args, n_devices),
           offsetof(struct kfd_ioctl_unmap_memory_from_gpu_args, n_success),
           (unsigned long)AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU);

    printf("alloc_flags:gtt=%#x writable=%#x executable=%#x aql_queue=%#x "
           "coherent=%#x uncached=%#x host_visible_coherent=%#x kernarg=%#x "
           "aql_profile=%#x executable_profile=%#x\n",
           KFD_IOC_ALLOC_MEM_FLAGS_GTT,
           KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE,
           KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE,
           KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM,
           KFD_IOC_ALLOC_MEM_FLAGS_COHERENT,
           KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
           KFD_IOC_ALLOC_MEM_FLAGS_GTT |
               KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
               KFD_IOC_ALLOC_MEM_FLAGS_COHERENT,
           KFD_IOC_ALLOC_MEM_FLAGS_GTT |
               KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
               KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
               KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
           KFD_IOC_ALLOC_MEM_FLAGS_GTT |
               KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
               KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
               KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
               KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM,
           KFD_IOC_ALLOC_MEM_FLAGS_GTT |
               KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
               KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
               KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE);

    printf("xnack:size=%zu align=%zu field=%zu request=%#lx query=-1 disabled=0 "
           "enabled=1\n",
           sizeof(struct kfd_ioctl_set_xnack_mode_args),
           _Alignof(struct kfd_ioctl_set_xnack_mode_args),
           offsetof(struct kfd_ioctl_set_xnack_mode_args, xnack_enabled),
           (unsigned long)AMDKFD_IOC_SET_XNACK_MODE);

    printf("smi_events:size=%zu align=%zu gpu_id=%zu anon_fd=%zu request=%#lx "
           "pre=%u post=%u mask=%#llx msg_size=%u\n",
           sizeof(struct kfd_ioctl_smi_events_args),
           _Alignof(struct kfd_ioctl_smi_events_args),
           offsetof(struct kfd_ioctl_smi_events_args, gpuid),
           offsetof(struct kfd_ioctl_smi_events_args, anon_fd),
           (unsigned long)AMDKFD_IOC_SMI_EVENTS,
           KFD_SMI_EVENT_GPU_PRE_RESET, KFD_SMI_EVENT_GPU_POST_RESET,
           (unsigned long long)(
               KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_GPU_PRE_RESET) |
               KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_GPU_POST_RESET)),
           KFD_SMI_EVENT_MSG_SIZE);
    return 0;
}
