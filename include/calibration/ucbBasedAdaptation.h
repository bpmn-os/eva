#pragma once

#include "../eva.h"

namespace EVA {

/**
 * @brief UCB-based weight adaptation strategy
 *
 * Recalculates weights after each offspring based on accumulated statistics (count, insertions, improvements).
 * Balances improvement (finding new best solutions), insertions (finding new good solutions),
 * and exploration (boosting underused operators using UCB-inspired term).
 *
 * @tparam Individual The individual type
 * @tparam Genome The genome type
 * @param improvementFactor Factor for improvement rate
 * @param insertionFactor Factor for insertion rate
 * @param explorationFactor Factor for exploration bonus (UCB-inspired)
 */
template <typename Individual, typename Genome = Individual>
std::function<void(EvolutionaryAlgorithm<Individual, Genome>*, const std::shared_ptr<const Individual>&, size_t, const Fitness&, bool, bool, bool)>
ucbBasedAdaptation(double improvementFactor = 0.7, double insertionFactor = 0.2, double explorationFactor = 0.1) {
  return [improvementFactor, insertionFactor, explorationFactor]([[maybe_unused]] EvolutionaryAlgorithm<Individual, Genome>* eva, [[maybe_unused]] const std::shared_ptr<const Individual>& offspring, size_t reproducer, [[maybe_unused]] const Fitness& fitness, bool isUnfit, bool isDuplicate, bool isFittest) {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;

    // Thread-local statistics (count, insertions, improvements) per operator
    thread_local std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> statistics;
    thread_local unsigned int totalCount = 0;

    // Initialize or reset if size changed
    if (statistics.size() != EVA::weights.size()) {
      statistics.resize(EVA::weights.size(), {0, 0, 0});
      totalCount = 0;
    }

    auto& [count, insertions, improvements] = statistics[reproducer];

    // Update statistics
    count++;
    totalCount++;
    if (!isUnfit && !isDuplicate) insertions++;
    if (isFittest) improvements++;

    // Exploitation: reward successful operators
    double insertionRate = (double)(insertions + 1) / (count + 1);
    double improvementRate = (double)(improvements + 1) / (count + 1);

    // Exploration: boost underused operators (UCB-inspired)
    double explorationTerm = sqrt(log(totalCount + 1) / (count + 1));

    double updatedWeight = improvementFactor * improvementRate
                         + insertionFactor * insertionRate
                         + explorationFactor * explorationTerm;
    EVA::totalWeight += updatedWeight - EVA::weights[reproducer];
    EVA::weights[reproducer] = updatedWeight;

    eva->normalizeWeights();
  };
}

} // namespace EVA
