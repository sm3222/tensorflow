/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_OPS_UTILS_H_
#define XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_OPS_UTILS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

enum class GPUCommunicationType {
  // The communication type could not be determined.
  UNDEFINED = 0,
  // Communication involves devices from multiple hosts, and every host
  // involved in the communication pattern has all of its devices participating.
  MULTI_HOST_WORLD_LEVEL = 1,
  // Communication involves devices from multiple hosts, but at least one of
  // the involved hosts has only a subset of its devices participating.
  MULTI_HOST_NON_WORLD_LEVEL = 2,
  // All devices participating in the collective operation reside on the same
  // fast-interconnect domain.
  SINGLE_PARTITION = 3
};

// Returns the type of communication pattern for a channel instruction.
absl::StatusOr<GPUCommunicationType> CommunicationType(
    int partition_size, const HloChannelInstruction& instr,
    const se::GpuComputeCapability& gpu_version);

// Enum to categorize collective-permute pairs based on whether they cross
// partition boundaries.
enum class PermuteType { kIntraPartition, kInterPartition };

// Properties of a single source-target pair in a collective-permute.
struct CollectivePermutePairProperty {
  PermuteType permute_type;
  int64_t source_idx;
  int64_t target_idx;
  int64_t num_devices_per_partition;
};

// Properties of a collective-permute instruction, categorizing its
// communication pattern.
struct CollectivePermuteProperty {
  // Categorizes pairs into intra- and inter-partition.
  absl::flat_hash_map<PermuteType, std::vector<CollectivePermutePairProperty>>
      permute_pairs_by_type;
  // True if at least one pair is inter-partition.
  bool has_inter_partition = false;
  // If 1, each device involved in the collective-permute either only sends or
  // only receives data. If 2, at least one device both sends and receives
  // data.
  int64_t num_device_edge = 1;
  // True if for every pair (s,t) in source_target_pairs, the pair (t,s) is
  // also present in source_target_pairs.
  bool is_all_mutual = false;
};

// Returns properties of collective-permute instruction based on its
// source-target pairs and number of devices per partition.
std::optional<CollectivePermuteProperty> GetCollectivePermuteProperty(
    const HloCollectivePermuteInstruction& instr,
    int64_t num_devices_per_partition);

// Enum to categorize collective-permute cost models based on communication
// patterns. The cost model is determined by the highest-latency pattern
// present in any device: TwoWayHasNonMutual > TwoWayAllMutual > OneWay.
enum class CollectivePermuteCostModelType {
  // Intra-partition: All devices only send or only receive data.
  kIntraPartitionOneWay,
  // Intra-partition: Devices send/receive, but only with the same peer
  // (e.g., {{0,1},{1,0}}).
  kIntraPartitionTwoWayAllMutual,
  // Intra-partition: At least one device sends to one peer and receives from
  // another (e.g., {{0,1},{1,2}}).
  kIntraPartitionTwoWayHasNonMutual,
  // Inter-partition: All devices only send or only receive data.
  kInterPartitionOneWay,
  // Inter-partition: Devices send/receive, but only with the same peer.
  kInterPartitionTwoWayAllMutual,
  // Inter-partition: At least one device sends to one peer and receives from
  // another.
  kInterPartitionTwoWayHasNonMutual,
};

// Returns cost model type based on collective-permute properties.
CollectivePermuteCostModelType GetCollectivePermuteCostModelType(
    const CollectivePermuteProperty& property);

// Returns true if instruction is a synchronous collective op.
bool IsGPUSyncCollective(const HloInstruction& instr);

// Returns true if all devices are within the same NVLink domain (slice).
bool IsIntraNVLinkDomain(const HloModuleConfig& config, int64_t slice_size);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_OPS_UTILS_H_
