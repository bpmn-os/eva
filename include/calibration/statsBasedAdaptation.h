#pragma once

#include "../eva.h"

namespace EVA {

/**
 * @brief Statistics-based weight adaptation strategy
 *
 * Recalculates weights after each offspring based on accumulated statistics (count, duplicates, improvements).
 * Balances improvement (finding better solutions) and novelty (avoiding duplicates).
 *
 * @tparam Individual The individual type
 * @tparam Genome The genome type
 * @param improvementFactor Factor for improvement rate
 * @param noveltyFactor Factor for novelty rate
 */
template <typename Individual, typename Genome = Individual>
std::function<void(EvolutionaryAlgorithm<Individual, Genome>*, const std::shared_ptr<const Individual>&, size_t, const Fitness&, bool, bool, bool)>
statsBasedAdaptation(double improvementFactor = 0.8, double insertionFactor = 0.2) {
  return [improvementFactor, insertionFactor]([[maybe_unused]] EvolutionaryAlgorithm<Individual, Genome>* eva, [[maybe_unused]] const std::shared_ptr<const Individual>& offspring, size_t reproducer, [[maybe_unused]] const Fitness& fitness, [[maybe_unused]] bool isUnfit, [[maybe_unused]] bool isDuplicate, [[maybe_unused]] bool isFittest) {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;

    // Update weight for this reproducer based on accumulated stats
    auto& [count, insertions, improvements] = EVA::stats[reproducer];

    // Calculate insertion rate
    double insertionRate = (double)(insertions+1) / (count+1);
    // Calculate improvement rate
    double improvementRate = (double)(improvements+1) / (count+1);

    // Weighted combination
    double updatedWeight = improvementFactor * improvementRate + insertionFactor * insertionRate;
    EVA::totalWeight += updatedWeight - EVA::weights[reproducer];
    EVA::weights[reproducer] = updatedWeight;

    // Normalize weights
    eva->normalizeWeights();
  };
}

} // namespace EVA
