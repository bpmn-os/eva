#pragma once

#include "../eva.h"

namespace EVA {

/**
 * @brief Improvement-based weight update strategy
 *
 * Recalculates weights after each offspring based on accumulated statistics (count, improvements).
 *
 * @tparam Individual The individual type
 * @tparam Genome The genome type
 */
template <typename Individual, typename Genome = Individual>
std::function<void(EvolutionaryAlgorithm<Individual, Genome>*, const std::shared_ptr<const Individual>&, size_t, const Fitness&, bool, bool, bool)>
improvementBasedAdaptation() {
  return []([[maybe_unused]] EvolutionaryAlgorithm<Individual, Genome>* eva, [[maybe_unused]] const std::shared_ptr<const Individual>& offspring, size_t reproducer, [[maybe_unused]] const Fitness& fitness, [[maybe_unused]] bool isUnfit, [[maybe_unused]] bool isDuplicate, bool isFittest) {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;

    // Thread-local statistics (count, improvements) per operator
    thread_local std::vector<std::tuple<unsigned int, unsigned int>> statistics;

    // Reset statistics when calibration is reset
    if (EVA::resetCalibration) {
      statistics.clear();
      statistics.resize(EVA::weights.size(), {0, 0});
      EVA::resetCalibration = false;
    }

    auto& [count, improvements] = statistics[reproducer];

    // Update statistics
    count++;
    if (isFittest) improvements++;

    // Calculate improvement rate
    double improvementRate = (double)(improvements + 1) / (count + 1);

    // Update weight
    double updatedWeight = improvementRate;
    EVA::totalWeight += updatedWeight - EVA::weights[reproducer];
    EVA::weights[reproducer] = updatedWeight;

    // Normalize weights
    eva->normalizeWeights();
  };
}

} // namespace EVA
