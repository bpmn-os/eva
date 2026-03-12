#pragma once

#include "../eva.h"

namespace EVA {

/**
 * @brief UCB-based weight adaptation strategy with exponential decay
 *
 * Recalculates weights after each offspring based on decayed statistics (count, insertions, improvements).
 * Balances improvement (finding new best solutions), insertions (finding new good solutions),
 * and exploration (boosting underused operators using UCB-inspired term).
 * Uses exponential decay to weight recent outcomes more heavily.
 *
 * @tparam Individual The individual type
 * @tparam Genome The genome type
 * @param decayFactor Decay for statistics (0.99 = ~100 effective window, 0.9 = ~10 effective window)
 * @param improvementFactor Factor for improvement rate
 * @param insertionFactor Factor for insertion rate
 * @param explorationFactor Factor for exploration bonus (UCB-inspired)
 */
template <typename Individual, typename Genome = Individual>
std::function<void(EvolutionaryAlgorithm<Individual, Genome>*, const std::shared_ptr<const Individual>&, size_t, const Fitness&, bool, bool, bool)>
decayingUcbBasedAdaptation(double decayFactor = 0.999, double improvementFactor = 0.7, double insertionFactor = 0.2, double explorationFactor = 0.1) {
  return [decayFactor, improvementFactor, insertionFactor, explorationFactor]([[maybe_unused]] EvolutionaryAlgorithm<Individual, Genome>* eva, [[maybe_unused]] const std::shared_ptr<const Individual>& offspring, size_t reproducer, [[maybe_unused]] const Fitness& fitness, bool isUnfit, bool isDuplicate, bool isFittest) {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;

    // Thread-local decayed statistics (count, insertions, improvements) per operator
    thread_local std::vector<std::tuple<double,double,double>> decayedStatistics;
    thread_local double totalDecayedCount = 0;

    // Initialize or reset if size changed
    if (decayedStatistics.size() != EVA::weights.size()) {
      decayedStatistics.resize(EVA::weights.size(), {0, 0, 0});
      totalDecayedCount = 0;
    }

    auto& [decayedCount, decayedInsertions, decayedImprovements] = decayedStatistics[reproducer];

    // Apply decay and update local statistics
    decayedCount = decayFactor * decayedCount + 1;
    decayedInsertions = decayFactor * decayedInsertions + (!isUnfit && !isDuplicate ? 1 : 0);
    decayedImprovements = decayFactor * decayedImprovements + (isFittest ? 1 : 0);

    // Update total count with decay (O(1) instead of O(n))
    totalDecayedCount = decayFactor * totalDecayedCount + 1;

    // Exploitation: reward successful operators
    double insertionRate = (decayedInsertions + 1) / (decayedCount + 1);
    double improvementRate = (decayedImprovements + 1) / (decayedCount + 1);

    // Exploration: boost underused operators (UCB-inspired)
    double explorationTerm = sqrt(log(totalDecayedCount + 1) / (decayedCount + 1));

    double updatedWeight = improvementFactor * improvementRate
                         + insertionFactor * insertionRate
                         + explorationFactor * explorationTerm;
    EVA::totalWeight += updatedWeight - EVA::weights[reproducer];
    EVA::weights[reproducer] = updatedWeight;

    eva->normalizeWeights();
  };
}

} // namespace EVA
