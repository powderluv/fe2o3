#include <linux/kfd_ioctl.h>
#include <stdint.h>
#include <stdio.h>

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

int main(void) {
    const uint64_t page = 4096;
    const uint64_t simd = 1216;
    const uint64_t simd_per_cu = 4;
    const uint64_t xcc = 8;
    const uint64_t array_count = 32;
    const uint64_t arrays_per_engine = 1;
    const uint64_t cu_per_xcc = simd / simd_per_cu / xcc;
    const uint64_t waves_per_xcc =
        cu_per_xcc * 40 < array_count / arrays_per_engine * 512
            ? cu_per_xcc * 40
            : array_count / arrays_per_engine * 512;
    const uint64_t control_stack =
        align_up(40 + waves_per_xcc * 8 + 8, page);
    const uint64_t workgroup_context =
        cu_per_xcc * (0x80000 + 0x4000 + 0x10000 + 0x1000);
    const uint64_t context_per_xcc =
        control_stack + align_up(workgroup_context, page);
    const uint64_t debug_per_xcc = align_up(waves_per_xcc * 32, 64);
    const uint64_t full_mapping =
        align_up((context_per_xcc + debug_per_xcc) * xcc, page);
    const uint32_t ring_flags =
        KFD_IOC_ALLOC_MEM_FLAGS_USERPTR |
        KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
        KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
        KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
    const uint32_t control_flags =
        KFD_IOC_ALLOC_MEM_FLAGS_USERPTR |
        KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
        KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
    const uint32_t eop_flags =
        KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
        KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
        KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;

    printf("queue_slots:min=0 max=1023 count=1024 zero_valid=1\n");
    printf("topology:simd=%llu simd_per_cu=%llu xcc=%llu arrays=%llu "
           "arrays_per_engine=%llu cu_per_xcc=%llu waves_per_xcc=%llu\n",
           (unsigned long long)simd, (unsigned long long)simd_per_cu,
           (unsigned long long)xcc, (unsigned long long)array_count,
           (unsigned long long)arrays_per_engine,
           (unsigned long long)cu_per_xcc,
           (unsigned long long)waves_per_xcc);
    printf("cwsr:ctl_per_xcc=%#llx wg_per_xcc=%#llx ctx_per_xcc=%#llx "
           "debug_per_xcc=%#llx mapping=%#llx\n",
           (unsigned long long)control_stack,
           (unsigned long long)workgroup_context,
           (unsigned long long)context_per_xcc,
           (unsigned long long)debug_per_xcc,
           (unsigned long long)full_mapping);
    printf("geometry:ring_min=4096 ring_max=2147483648 packet=64 "
           "counter=8 eop=4096 cwsr_kfd_align=4096 "
           "cwsr_rocr_svm_align=2097152 doorbell=8 doorbell_slice=8192\n");
    printf("rocr_flags:ring=%#x control=%#x eop=%#x\n",
           ring_flags, control_flags, eop_flags);
    return 0;
}
