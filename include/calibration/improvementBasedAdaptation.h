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
std::function<void(EvolutionaryAlgorithm<Individual, Genome>*, const std::shared_ptr<const Individual>&, size_t, const Fitness&, bool, bool)>
improvementBasedAdaptation() {
  return []([[maybe_unused]] EvolutionaryAlgorithm<Individual, Genome>* eva, [[maybe_unused]] const std::shared_ptr<const Individual>& offspring, size_t reproducer, [[maybe_unused]] const Fitness& fitness, [[maybe_unused]] bool isDuplicate, [[maybe_unused]] bool isFittest) {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;
    // Update weight for this reproducer based on accumulated stats
    auto& [count, duplicates, improvements] = EVA::stats[reproducer];

    // Calculate improvement rate
    double improvementRate = (double)(improvements+1) / (count+1);

    // Weighted combination
    double updatedWeight = improvementRate;
    EVA::totalWeight += updatedWeight - EVA::weights[reproducer];
    EVA::weights[reproducer] = updatedWeight;

    // Normalize weights
    eva->normalizeWeights();
  };
}

} // namespace EVA
