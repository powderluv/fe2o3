#ifndef FE2O3_HSA_RUNTIME_H
#define FE2O3_HSA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#define FE2O3_HSA_AGENT_CAPACITY 64
#define FE2O3_HSA_POOL_CAPACITY 256
#define FE2O3_HSA_TEXT_CAPACITY 128

typedef struct Fe2o3HsaAgentRecord {
  uint64_t handle;
  uint32_t node;
  uint32_t device_type;
  uint32_t feature;
  uint32_t profile;
  uint32_t queue_min_size;
  uint32_t queue_max_size;
  uint32_t queue_type;
  uint32_t domain;
  uint32_t bdf_id;
  char name[64];
  char uuid[22];
  char isa[FE2O3_HSA_TEXT_CAPACITY];
  uint32_t matching_isa_count;
} Fe2o3HsaAgentRecord;

typedef struct Fe2o3HsaPoolRecord {
  uint64_t handle;
  uint64_t owner_agent;
  uint32_t owner_node;
  uint32_t segment;
  uint32_t global_flags;
  uint32_t runtime_alloc_allowed;
  uint64_t runtime_alloc_alignment;
} Fe2o3HsaPoolRecord;

typedef struct Fe2o3HipDeviceRecord {
  uint8_t uuid[16];
  char pci_bus_id[32];
  int32_t round_trip_ordinal;
} Fe2o3HipDeviceRecord;

typedef struct Fe2o3HsaSymbolRecord {
  uint64_t handle;
  uint64_t kernel_object;
  uint32_t kind;
  uint32_t kernarg_size;
  uint32_t kernarg_alignment;
  uint32_t group_segment_size;
  uint32_t private_segment_size;
  char name[FE2O3_HSA_TEXT_CAPACITY];
} Fe2o3HsaSymbolRecord;

typedef struct Fe2o3HsaQueueRecord {
  uintptr_t pointer;
  uint64_t id;
  uint32_t size;
  uintptr_t async_error;
} Fe2o3HsaQueueRecord;

typedef struct Fe2o3HsaDispatchTimeRecord {
  uint64_t start;
  uint64_t end;
} Fe2o3HsaDispatchTimeRecord;

_Static_assert(sizeof(Fe2o3HsaAgentRecord) == 264,
               "Fe2o3HsaAgentRecord ABI size");
_Static_assert(_Alignof(Fe2o3HsaAgentRecord) == 8,
               "Fe2o3HsaAgentRecord ABI alignment");
_Static_assert(sizeof(Fe2o3HsaPoolRecord) == 40, "Fe2o3HsaPoolRecord ABI size");
_Static_assert(_Alignof(Fe2o3HsaPoolRecord) == 8,
               "Fe2o3HsaPoolRecord ABI alignment");
_Static_assert(sizeof(Fe2o3HipDeviceRecord) == 52,
               "Fe2o3HipDeviceRecord ABI size");
_Static_assert(_Alignof(Fe2o3HipDeviceRecord) == 4,
               "Fe2o3HipDeviceRecord ABI alignment");
_Static_assert(sizeof(Fe2o3HsaSymbolRecord) == 168,
               "Fe2o3HsaSymbolRecord ABI size");
_Static_assert(_Alignof(Fe2o3HsaSymbolRecord) == 8,
               "Fe2o3HsaSymbolRecord ABI alignment");
_Static_assert(sizeof(Fe2o3HsaQueueRecord) == 32,
               "Fe2o3HsaQueueRecord ABI size");
_Static_assert(_Alignof(Fe2o3HsaQueueRecord) == 8,
               "Fe2o3HsaQueueRecord ABI alignment");
_Static_assert(sizeof(Fe2o3HsaDispatchTimeRecord) == 16,
               "Fe2o3HsaDispatchTimeRecord ABI size");
_Static_assert(_Alignof(Fe2o3HsaDispatchTimeRecord) == 8,
               "Fe2o3HsaDispatchTimeRecord ABI alignment");

int32_t fe2o3_hsa_init(void);
int32_t fe2o3_hsa_shut_down(void);
int32_t fe2o3_hsa_runtime_version(uint16_t *major, uint16_t *minor);
uintptr_t fe2o3_hsa_runtime_function_address(void);
uintptr_t fe2o3_hip_runtime_function_address(void);
int32_t fe2o3_hsa_collect_agents(Fe2o3HsaAgentRecord *records,
                                 uint32_t capacity, uint32_t *count);
int32_t fe2o3_hsa_collect_kernarg_pools(Fe2o3HsaPoolRecord *records,
                                        uint32_t capacity, uint32_t *count);
int32_t fe2o3_hip_observe_device(int32_t ordinal, Fe2o3HipDeviceRecord *record);

int32_t fe2o3_hsa_reader_create(const void *bytes, size_t len,
                                uint64_t *reader);
int32_t fe2o3_hsa_reader_destroy(uint64_t reader);
int32_t fe2o3_hsa_executable_create(uint32_t profile, uint64_t *executable);
int32_t fe2o3_hsa_executable_load(uint64_t executable, uint64_t agent,
                                  uint64_t reader, uint64_t *loaded);
int32_t fe2o3_hsa_executable_freeze(uint64_t executable);
int32_t fe2o3_hsa_executable_destroy(uint64_t executable);
int32_t fe2o3_hsa_resolve_symbol(uint64_t executable, uint64_t agent,
                                 const char *name,
                                 Fe2o3HsaSymbolRecord *record);

int32_t fe2o3_hsa_pool_allocate(uint64_t pool, size_t len, void **address);
int32_t fe2o3_hsa_allow_access(uint64_t agent, void *address);
int32_t fe2o3_hsa_memory_free(void *address);
int32_t fe2o3_hsa_queue_create(uint64_t agent, uint32_t size,
                               Fe2o3HsaQueueRecord *record);
int32_t fe2o3_hsa_queue_async_error(const Fe2o3HsaQueueRecord *record);
int32_t fe2o3_hsa_queue_enable_profiling(
    const Fe2o3HsaQueueRecord *record);
int32_t fe2o3_hsa_queue_destroy(Fe2o3HsaQueueRecord *record);
int32_t fe2o3_hsa_signal_create(int64_t initial_value, uint64_t *signal);
int32_t fe2o3_hsa_signal_destroy(uint64_t signal);
int64_t fe2o3_hsa_signal_load_acquire(uint64_t signal);
int32_t fe2o3_hsa_signal_store_release(uint64_t signal, int64_t value);
int32_t fe2o3_hsa_system_timestamp_frequency(uint64_t *frequency);
int32_t fe2o3_hsa_dispatch_time(uint64_t agent, uint64_t signal,
                                Fe2o3HsaDispatchTimeRecord *record);
int32_t
fe2o3_hsa_test_malformed_queue_destroy_failure(Fe2o3HsaQueueRecord *record);
void fe2o3_hsa_test_release_malformed_queue_record(Fe2o3HsaQueueRecord *record);
int32_t fe2o3_hsa_publish_kernel_dispatch(
    const Fe2o3HsaQueueRecord *queue, const uint32_t grid[3],
    const uint32_t workgroup[3], uint32_t private_segment_size,
    uint32_t group_segment_size, uint64_t kernel_object, void *kernarg,
    uint64_t completion_signal, uint64_t *packet_id);

#endif
