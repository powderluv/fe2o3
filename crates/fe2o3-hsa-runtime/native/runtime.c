#include "runtime.h"

#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct AgentCollection {
  Fe2o3HsaAgentRecord *records;
  uint32_t capacity;
  uint32_t count;
} AgentCollection;

typedef struct IsaCollection {
  Fe2o3HsaAgentRecord *record;
} IsaCollection;

typedef struct PoolCollection {
  Fe2o3HsaPoolRecord *records;
  uint32_t capacity;
  uint32_t count;
  hsa_agent_t owner;
  uint32_t owner_node;
} PoolCollection;

static hsa_status_t collect_isa(hsa_isa_t isa, void *opaque) {
  IsaCollection *collection = (IsaCollection *)opaque;
  uint32_t length = 0;
  hsa_status_t status =
      hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, (void *)&length);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  if (length == 0 || length >= FE2O3_HSA_TEXT_CAPACITY)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  char name[FE2O3_HSA_TEXT_CAPACITY] = {0};
  status = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, (void *)name);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  static const char gfx942[] = "amdgcn-amd-amdhsa--gfx942";
  static const char gfx950[] = "amdgcn-amd-amdhsa--gfx950";
  const bool reviewed_gfx942 =
      strncmp(name, gfx942, sizeof(gfx942) - 1) == 0 &&
      (name[sizeof(gfx942) - 1] == '\0' || name[sizeof(gfx942) - 1] == ':');
  const bool reviewed_gfx950 =
      strncmp(name, gfx950, sizeof(gfx950) - 1) == 0 &&
      (name[sizeof(gfx950) - 1] == '\0' || name[sizeof(gfx950) - 1] == ':');
  if (reviewed_gfx942 || reviewed_gfx950) {
    collection->record->matching_isa_count++;
    if (collection->record->matching_isa_count == 1)
      memcpy(collection->record->isa, name, (size_t)length);
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t collect_agent(hsa_agent_t agent, void *opaque) {
  AgentCollection *collection = (AgentCollection *)opaque;
  if (collection->count == collection->capacity)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  Fe2o3HsaAgentRecord *record = &collection->records[collection->count];
  memset(record, 0, sizeof(*record));
  record->handle = agent.handle;
#define GET_AGENT_INFO(attribute, field)                                       \
  do {                                                                         \
    hsa_status_t info_status =                                                 \
        hsa_agent_get_info(agent, (hsa_agent_info_t)(attribute), (field));     \
    if (info_status != HSA_STATUS_SUCCESS)                                     \
      return info_status;                                                      \
  } while (0)
  GET_AGENT_INFO(HSA_AGENT_INFO_NODE, &record->node);
  GET_AGENT_INFO(HSA_AGENT_INFO_DEVICE, &record->device_type);
  GET_AGENT_INFO(HSA_AGENT_INFO_FEATURE, &record->feature);
  GET_AGENT_INFO(HSA_AGENT_INFO_PROFILE, &record->profile);
  GET_AGENT_INFO(HSA_AGENT_INFO_QUEUE_MIN_SIZE, &record->queue_min_size);
  GET_AGENT_INFO(HSA_AGENT_INFO_QUEUE_MAX_SIZE, &record->queue_max_size);
  GET_AGENT_INFO(HSA_AGENT_INFO_QUEUE_TYPE, &record->queue_type);
  GET_AGENT_INFO(HSA_AGENT_INFO_NAME, record->name);
  GET_AGENT_INFO(HSA_AMD_AGENT_INFO_DOMAIN, &record->domain);
  GET_AGENT_INFO(HSA_AMD_AGENT_INFO_BDFID, &record->bdf_id);
  GET_AGENT_INFO(HSA_AMD_AGENT_INFO_UUID, record->uuid);
#undef GET_AGENT_INFO

  IsaCollection isas = {.record = record};
  hsa_status_t status = hsa_agent_iterate_isas(agent, collect_isa, &isas);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  collection->count++;
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t collect_pool(hsa_amd_memory_pool_t pool, void *opaque) {
  PoolCollection *collection = (PoolCollection *)opaque;
  if (collection->count == collection->capacity)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  Fe2o3HsaPoolRecord *record = &collection->records[collection->count];
  memset(record, 0, sizeof(*record));
  record->handle = pool.handle;
  record->owner_agent = collection->owner.handle;
  record->owner_node = collection->owner_node;
  bool alloc_allowed = false;
#define GET_POOL_INFO(attribute, field)                                        \
  do {                                                                         \
    hsa_status_t info_status =                                                 \
        hsa_amd_memory_pool_get_info(pool, (attribute), (field));              \
    if (info_status != HSA_STATUS_SUCCESS)                                     \
      return info_status;                                                      \
  } while (0)
  GET_POOL_INFO(HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &record->segment);
  if (record->segment == HSA_AMD_SEGMENT_GLOBAL)
    GET_POOL_INFO(HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &record->global_flags);
  GET_POOL_INFO(HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
  record->runtime_alloc_allowed = alloc_allowed ? 1U : 0U;
  if (alloc_allowed)
    GET_POOL_INFO(HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT,
                  &record->runtime_alloc_alignment);
#undef GET_POOL_INFO
  collection->count++;
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t collect_agent_pools(hsa_agent_t agent, void *opaque) {
  PoolCollection *collection = (PoolCollection *)opaque;
  uint32_t node = 0;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &node);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  collection->owner = agent;
  collection->owner_node = node;
  return hsa_amd_agent_iterate_memory_pools(agent, collect_pool, collection);
}

int32_t fe2o3_hsa_init(void) { return (int32_t)hsa_init(); }

int32_t fe2o3_hsa_shut_down(void) { return (int32_t)hsa_shut_down(); }

int32_t fe2o3_hsa_runtime_version(uint16_t *major, uint16_t *minor) {
  if (major == NULL || minor == NULL)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  hsa_status_t status =
      hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, major);
  if (status != HSA_STATUS_SUCCESS)
    return (int32_t)status;
  return (int32_t)hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, minor);
}

uintptr_t fe2o3_hsa_runtime_function_address(void) {
  return (uintptr_t)&hsa_init;
}

uintptr_t fe2o3_hip_runtime_function_address(void) {
  return (uintptr_t)&hipInit;
}

int32_t fe2o3_hsa_collect_agents(Fe2o3HsaAgentRecord *records,
                                 uint32_t capacity, uint32_t *count) {
  if (records == NULL || count == NULL || capacity == 0)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  AgentCollection collection = {
      .records = records, .capacity = capacity, .count = 0};
  hsa_status_t status = hsa_iterate_agents(collect_agent, &collection);
  *count = collection.count;
  return (int32_t)status;
}

int32_t fe2o3_hsa_collect_kernarg_pools(Fe2o3HsaPoolRecord *records,
                                        uint32_t capacity, uint32_t *count) {
  if (records == NULL || count == NULL || capacity == 0)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  PoolCollection collection = {
      .records = records, .capacity = capacity, .count = 0};
  hsa_status_t status = hsa_iterate_agents(collect_agent_pools, &collection);
  *count = collection.count;
  return (int32_t)status;
}

int32_t fe2o3_hip_observe_device(int32_t ordinal,
                                 Fe2o3HipDeviceRecord *record) {
  if (ordinal < 0 || record == NULL)
    return (int32_t)hipErrorInvalidValue;
  memset(record, 0, sizeof(*record));
  hipError_t status = hipInit(0);
  if (status != hipSuccess)
    return (int32_t)status;
  hipUUID uuid = {{0}};
  status = hipDeviceGetUuid(&uuid, ordinal);
  if (status != hipSuccess)
    return (int32_t)status;
  memcpy(record->uuid, uuid.bytes, sizeof(record->uuid));
  status = hipDeviceGetPCIBusId(record->pci_bus_id,
                                (int)sizeof(record->pci_bus_id), ordinal);
  if (status != hipSuccess)
    return (int32_t)status;
  status =
      hipDeviceGetByPCIBusId(&record->round_trip_ordinal, record->pci_bus_id);
  return (int32_t)status;
}

int32_t fe2o3_hsa_reader_create(const void *bytes, size_t len,
                                uint64_t *reader) {
  if (reader == NULL)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  hsa_code_object_reader_t value = {0};
  hsa_status_t status =
      hsa_code_object_reader_create_from_memory(bytes, len, &value);
  *reader = value.handle;
  return (int32_t)status;
}

int32_t fe2o3_hsa_reader_destroy(uint64_t reader) {
  return (int32_t)hsa_code_object_reader_destroy(
      (hsa_code_object_reader_t){.handle = reader});
}

int32_t fe2o3_hsa_executable_create(uint32_t profile, uint64_t *executable) {
  if (executable == NULL)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  hsa_executable_t value = {0};
  hsa_status_t status = hsa_executable_create_alt(
      (hsa_profile_t)profile, HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR, NULL,
      &value);
  *executable = value.handle;
  return (int32_t)status;
}

int32_t fe2o3_hsa_executable_load(uint64_t executable, uint64_t agent,
                                  uint64_t reader, uint64_t *loaded) {
  if (loaded == NULL)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  hsa_loaded_code_object_t value = {0};
  hsa_status_t status = hsa_executable_load_agent_code_object(
      (hsa_executable_t){.handle = executable}, (hsa_agent_t){.handle = agent},
      (hsa_code_object_reader_t){.handle = reader}, NULL, &value);
  *loaded = value.handle;
  return (int32_t)status;
}

int32_t fe2o3_hsa_executable_freeze(uint64_t executable) {
  return (int32_t)hsa_executable_freeze(
      (hsa_executable_t){.handle = executable}, NULL);
}

int32_t fe2o3_hsa_executable_destroy(uint64_t executable) {
  return (int32_t)hsa_executable_destroy(
      (hsa_executable_t){.handle = executable});
}

int32_t fe2o3_hsa_resolve_symbol(uint64_t executable, uint64_t agent,
                                 const char *name,
                                 Fe2o3HsaSymbolRecord *record) {
  if (name == NULL || record == NULL)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  memset(record, 0, sizeof(*record));
  hsa_agent_t hsa_agent = {.handle = agent};
  hsa_executable_symbol_t symbol = {0};
  hsa_status_t status = hsa_executable_get_symbol_by_name(
      (hsa_executable_t){.handle = executable}, name, &hsa_agent, &symbol);
  if (status != HSA_STATUS_SUCCESS)
    return (int32_t)status;
  record->handle = symbol.handle;
#define GET_SYMBOL_INFO(attribute, field)                                      \
  do {                                                                         \
    status = hsa_executable_symbol_get_info(symbol, (attribute), (field));     \
    if (status != HSA_STATUS_SUCCESS)                                          \
      return (int32_t)status;                                                  \
  } while (0)
  uint32_t name_length = 0;
  GET_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &record->kind);
  GET_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length);
  if (name_length == 0 || name_length >= FE2O3_HSA_TEXT_CAPACITY)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  GET_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_NAME, record->name);
  GET_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                  &record->kernel_object);
  GET_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                  &record->kernarg_size);
  GET_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
                  &record->kernarg_alignment);
  GET_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                  &record->group_segment_size);
  GET_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                  &record->private_segment_size);
#undef GET_SYMBOL_INFO
  return (int32_t)HSA_STATUS_SUCCESS;
}

int32_t fe2o3_hsa_pool_allocate(uint64_t pool, size_t len, void **address) {
  return (int32_t)hsa_amd_memory_pool_allocate(
      (hsa_amd_memory_pool_t){.handle = pool}, len, 0, address);
}

int32_t fe2o3_hsa_allow_access(uint64_t agent, void *address) {
  hsa_agent_t value = {.handle = agent};
  return (int32_t)hsa_amd_agents_allow_access(1, &value, NULL, address);
}

int32_t fe2o3_hsa_memory_free(void *address) {
  return (int32_t)hsa_amd_memory_pool_free(address);
}

static void queue_error_callback(hsa_status_t status, hsa_queue_t *source,
                                 void *data) {
  (void)source;
  _Atomic int32_t *first_error = (_Atomic int32_t *)data;
  int32_t expected = (int32_t)HSA_STATUS_SUCCESS;
  (void)atomic_compare_exchange_strong(first_error, &expected, (int32_t)status);
}

typedef hsa_status_t (*Fe2o3QueueDestroyFn)(hsa_queue_t *queue);

static int32_t cleanup_malformed_queue(hsa_queue_t *queue,
                                       _Atomic int32_t *async_error,
                                       Fe2o3HsaQueueRecord *record,
                                       Fe2o3QueueDestroyFn destroy) {
  if (queue == NULL) {
    free(async_error);
    return (int32_t)HSA_STATUS_ERROR_INVALID_QUEUE;
  }
  hsa_status_t cleanup = destroy(queue);
  if (cleanup == HSA_STATUS_SUCCESS) {
    free(async_error);
    return (int32_t)HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  // Destruction is ambiguous. Retain both allocations and return their exact
  // authority to Rust, which treats this nonzero record as terminal.
  record->pointer = (uintptr_t)queue;
  record->id = queue->id;
  record->size = queue->size;
  record->async_error = (uintptr_t)async_error;
  return (int32_t)cleanup;
}

int32_t fe2o3_hsa_queue_create(uint64_t agent, uint32_t size,
                               Fe2o3HsaQueueRecord *record) {
  if (record == NULL)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  memset(record, 0, sizeof(*record));
  _Atomic int32_t *async_error = calloc(1, sizeof(*async_error));
  if (async_error == NULL)
    return (int32_t)HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  hsa_queue_t *queue = NULL;
  hsa_status_t status = hsa_queue_create(
      (hsa_agent_t){.handle = agent}, size, HSA_QUEUE_TYPE_SINGLE,
      queue_error_callback, async_error, UINT32_MAX, UINT32_MAX, &queue);
  if (status != HSA_STATUS_SUCCESS) {
    free(async_error);
    return (int32_t)status;
  }
  if (queue == NULL || queue->size == 0 ||
      (queue->size & (queue->size - 1)) != 0) {
    return cleanup_malformed_queue(queue, async_error, record,
                                   hsa_queue_destroy);
  }
  record->pointer = (uintptr_t)queue;
  record->id = queue->id;
  record->size = queue->size;
  record->async_error = (uintptr_t)async_error;
  return (int32_t)HSA_STATUS_SUCCESS;
}

int32_t fe2o3_hsa_queue_async_error(const Fe2o3HsaQueueRecord *record) {
  if (record == NULL || record->async_error == 0)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  _Atomic int32_t *async_error = (_Atomic int32_t *)record->async_error;
  return atomic_load(async_error);
}

int32_t fe2o3_hsa_queue_enable_profiling(
    const Fe2o3HsaQueueRecord *record) {
  if (record == NULL || record->pointer == 0)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return (int32_t)hsa_amd_profiling_set_profiler_enabled(
      (hsa_queue_t *)record->pointer, 1);
}

int32_t fe2o3_hsa_queue_destroy(Fe2o3HsaQueueRecord *record) {
  if (record == NULL || record->pointer == 0 || record->async_error == 0)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  hsa_status_t status = hsa_queue_destroy((hsa_queue_t *)record->pointer);
  if (status == HSA_STATUS_SUCCESS) {
    free((void *)record->async_error);
    memset(record, 0, sizeof(*record));
  }
  return (int32_t)status;
}

int32_t fe2o3_hsa_signal_create(int64_t initial_value, uint64_t *signal) {
  if (signal == NULL)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  hsa_signal_t value = {0};
  hsa_status_t status =
      hsa_signal_create((hsa_signal_value_t)initial_value, 0, NULL, &value);
  *signal = value.handle;
  return (int32_t)status;
}

int32_t fe2o3_hsa_signal_destroy(uint64_t signal) {
  return (int32_t)hsa_signal_destroy((hsa_signal_t){.handle = signal});
}

int64_t fe2o3_hsa_signal_load_acquire(uint64_t signal) {
  return (int64_t)hsa_signal_load_scacquire((hsa_signal_t){.handle = signal});
}

int32_t fe2o3_hsa_signal_store_release(uint64_t signal, int64_t value) {
  if (signal == 0)
    return (int32_t)HSA_STATUS_ERROR_INVALID_SIGNAL;
  hsa_signal_store_screlease((hsa_signal_t){.handle = signal},
                             (hsa_signal_value_t)value);
  return (int32_t)HSA_STATUS_SUCCESS;
}

int32_t fe2o3_hsa_system_timestamp_frequency(uint64_t *frequency) {
  if (frequency == NULL)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return (int32_t)hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                                      frequency);
}

int32_t fe2o3_hsa_dispatch_time(uint64_t agent, uint64_t signal,
                                Fe2o3HsaDispatchTimeRecord *record) {
  if (agent == 0 || signal == 0 || record == NULL)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  hsa_amd_profiling_dispatch_time_t observed = {0};
  hsa_status_t status = hsa_amd_profiling_get_dispatch_time(
      (hsa_agent_t){.handle = agent}, (hsa_signal_t){.handle = signal},
      &observed);
  if (status == HSA_STATUS_SUCCESS) {
    record->start = observed.start;
    record->end = observed.end;
  }
  return (int32_t)status;
}

static hsa_status_t fe2o3_test_queue_destroy_failure(hsa_queue_t *queue) {
  (void)queue;
  return HSA_STATUS_ERROR_EXCEPTION;
}

int32_t
fe2o3_hsa_test_malformed_queue_destroy_failure(Fe2o3HsaQueueRecord *record) {
  if (record == NULL)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  memset(record, 0, sizeof(*record));
  hsa_queue_t *queue = calloc(1, sizeof(*queue));
  _Atomic int32_t *async_error = calloc(1, sizeof(*async_error));
  if (queue == NULL || async_error == NULL) {
    free(queue);
    free(async_error);
    return (int32_t)HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  queue->id = 0;
  queue->size = 3;
  return cleanup_malformed_queue(queue, async_error, record,
                                 fe2o3_test_queue_destroy_failure);
}

void fe2o3_hsa_test_release_malformed_queue_record(
    Fe2o3HsaQueueRecord *record) {
  if (record == NULL)
    return;
  free((void *)record->pointer);
  free((void *)record->async_error);
  memset(record, 0, sizeof(*record));
}

int32_t fe2o3_hsa_publish_kernel_dispatch(
    const Fe2o3HsaQueueRecord *queue_record, const uint32_t grid[3],
    const uint32_t workgroup[3], uint32_t private_segment_size,
    uint32_t group_segment_size, uint64_t kernel_object, void *kernarg,
    uint64_t completion_signal, uint64_t *packet_id) {
  if (queue_record == NULL || queue_record->pointer == 0 || grid == NULL ||
      workgroup == NULL || kernarg == NULL || completion_signal == 0 ||
      packet_id == NULL)
    return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (uint32_t index = 0; index != 3; ++index) {
    if (grid[index] == 0 || workgroup[index] == 0 ||
        workgroup[index] > UINT16_MAX || grid[index] < workgroup[index])
      return (int32_t)HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_queue_t *queue = (hsa_queue_t *)queue_record->pointer;
  uint64_t id = hsa_queue_add_write_index_relaxed(queue, 1);
  while (id - hsa_queue_load_read_index_scacquire(queue) >= queue->size) {
  }
  hsa_kernel_dispatch_packet_t *packet =
      &((hsa_kernel_dispatch_packet_t *)
            queue->base_address)[id & (queue->size - 1)];
  _Static_assert(offsetof(hsa_kernel_dispatch_packet_t, header) == 0,
                 "AQL header offset");
  _Static_assert(offsetof(hsa_kernel_dispatch_packet_t, setup) == 2,
                 "AQL setup offset");
  memset(packet, 0, sizeof(*packet));
  packet->workgroup_size_x = (uint16_t)workgroup[0];
  packet->workgroup_size_y = (uint16_t)workgroup[1];
  packet->workgroup_size_z = (uint16_t)workgroup[2];
  packet->grid_size_x = grid[0];
  packet->grid_size_y = grid[1];
  packet->grid_size_z = grid[2];
  packet->private_segment_size = private_segment_size;
  packet->group_segment_size = group_segment_size;
  packet->kernel_object = kernel_object;
  packet->kernarg_address = kernarg;
  packet->completion_signal.handle = completion_signal;

  uint16_t dimensions = grid[2] > 1 ? 3U : (grid[1] > 1 ? 2U : 1U);
  uint16_t setup =
      (uint16_t)(dimensions << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
  uint16_t header =
      (uint16_t)(HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (uint16_t)(HSA_FENCE_SCOPE_SYSTEM
                 << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (uint16_t)(HSA_FENCE_SCOPE_SYSTEM
                 << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  uint32_t full_header = (uint32_t)header | ((uint32_t)setup << 16);
  atomic_store_explicit((_Atomic uint32_t *)&packet->header, full_header,
                        memory_order_release);
  hsa_signal_store_screlease(queue->doorbell_signal, (hsa_signal_value_t)id);
  *packet_id = id;
  return (int32_t)HSA_STATUS_SUCCESS;
}
