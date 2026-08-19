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

    printf("create_queue:size=%zu align=%zu ring_base=%zu write_pointer=%zu "
           "read_pointer=%zu doorbell_offset=%zu ring_size=%zu gpu_id=%zu "
           "queue_type=%zu queue_percentage=%zu queue_priority=%zu queue_id=%zu "
           "eop_address=%zu eop_size=%zu ctx_address=%zu ctx_size=%zu "
           "ctl_stack_size=%zu sdma_engine_id=%zu pad=%zu request=%#lx\n",
           sizeof(struct kfd_ioctl_create_queue_args),
           _Alignof(struct kfd_ioctl_create_queue_args),
           offsetof(struct kfd_ioctl_create_queue_args, ring_base_address),
           offsetof(struct kfd_ioctl_create_queue_args, write_pointer_address),
           offsetof(struct kfd_ioctl_create_queue_args, read_pointer_address),
           offsetof(struct kfd_ioctl_create_queue_args, doorbell_offset),
           offsetof(struct kfd_ioctl_create_queue_args, ring_size),
           offsetof(struct kfd_ioctl_create_queue_args, gpu_id),
           offsetof(struct kfd_ioctl_create_queue_args, queue_type),
           offsetof(struct kfd_ioctl_create_queue_args, queue_percentage),
           offsetof(struct kfd_ioctl_create_queue_args, queue_priority),
           offsetof(struct kfd_ioctl_create_queue_args, queue_id),
           offsetof(struct kfd_ioctl_create_queue_args, eop_buffer_address),
           offsetof(struct kfd_ioctl_create_queue_args, eop_buffer_size),
           offsetof(struct kfd_ioctl_create_queue_args, ctx_save_restore_address),
           offsetof(struct kfd_ioctl_create_queue_args, ctx_save_restore_size),
           offsetof(struct kfd_ioctl_create_queue_args, ctl_stack_size),
           offsetof(struct kfd_ioctl_create_queue_args, sdma_engine_id),
           offsetof(struct kfd_ioctl_create_queue_args, pad),
           (unsigned long)AMDKFD_IOC_CREATE_QUEUE);

    printf("destroy_queue:size=%zu align=%zu queue_id=%zu pad=%zu request=%#lx\n",
           sizeof(struct kfd_ioctl_destroy_queue_args),
           _Alignof(struct kfd_ioctl_destroy_queue_args),
           offsetof(struct kfd_ioctl_destroy_queue_args, queue_id),
           offsetof(struct kfd_ioctl_destroy_queue_args, pad),
           (unsigned long)AMDKFD_IOC_DESTROY_QUEUE);

    printf("update_queue:size=%zu align=%zu ring_base=%zu queue_id=%zu "
           "ring_size=%zu queue_percentage=%zu queue_priority=%zu request=%#lx\n",
           sizeof(struct kfd_ioctl_update_queue_args),
           _Alignof(struct kfd_ioctl_update_queue_args),
           offsetof(struct kfd_ioctl_update_queue_args, ring_base_address),
           offsetof(struct kfd_ioctl_update_queue_args, queue_id),
           offsetof(struct kfd_ioctl_update_queue_args, ring_size),
           offsetof(struct kfd_ioctl_update_queue_args, queue_percentage),
           offsetof(struct kfd_ioctl_update_queue_args, queue_priority),
           (unsigned long)AMDKFD_IOC_UPDATE_QUEUE);

    printf("queue_constants:compute_aql=%#x max_percentage=%u max_priority=%u "
           "min_ring_size=%u\n",
           KFD_IOC_QUEUE_TYPE_COMPUTE_AQL,
           KFD_MAX_QUEUE_PERCENTAGE,
           KFD_MAX_QUEUE_PRIORITY,
           KFD_MIN_QUEUE_RING_SIZE);

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
