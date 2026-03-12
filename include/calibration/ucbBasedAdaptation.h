#pragma once

#include "../eva.h"

namespace EVA {

/**
 * @brief UCB-based weight adaptation strategy
 *
 * Recalculates weights after each offspring based on statistics (count, insertions, improvements).
 * Balances improvement (finding new best solutions), insertions (finding new good solutions),
 * and exploration (boosting underused operators using UCB-inspired term).
 * Uses exponential decay to weight recent outcomes more heavily (1.0 = no decay).
 *
 * @tparam Individual The individual type
 * @tparam Genome The genome type
 * @param decayFactor Decay for statistics (1.0 = no decay)
 * @param improvementFactor Factor for improvement rate
 * @param insertionFactor Factor for insertion rate
 * @param explorationFactor Factor for exploration bonus (UCB-inspired)
 */
template <typename Individual, typename Genome = Individual>
std::function<void(EvolutionaryAlgorithm<Individual, Genome>*, const std::shared_ptr<const Individual>&, size_t, const Fitness&, bool, bool, bool)>
ucbBasedAdaptation(double decayFactor = 0.999, double improvementFactor = 0.7, double insertionFactor = 0.2, double explorationFactor = 0.1) {
  return [decayFactor, improvementFactor, insertionFactor, explorationFactor]([[maybe_unused]] EvolutionaryAlgorithm<Individual, Genome>* eva, [[maybe_unused]] const std::shared_ptr<const Individual>& offspring, size_t reproducer, [[maybe_unused]] const Fitness& fitness, bool isUnfit, bool isDuplicate, bool isFittest) {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;

    // Thread-local statistics (count, insertions, improvements) per operator
    thread_local std::vector<std::tuple<double,double,double>> statistics;
    thread_local double totalCount = 0;

    // Reset statistics when calibration is reset
    if (EVA::resetCalibration) {
      statistics.clear();
      statistics.resize(EVA::weights.size(), {0, 0, 0});
      totalCount = 0;
      EVA::resetCalibration = false;
    }

    auto& [count, insertions, improvements] = statistics[reproducer];

    // Apply decay and update local statistics
    count = decayFactor * count + 1;
    insertions = decayFactor * insertions + (!isUnfit && !isDuplicate ? 1 : 0);
    improvements = decayFactor * improvements + (isFittest ? 1 : 0);

    // Update total count with decay (O(1) instead of O(n))
    totalCount = decayFactor * totalCount + 1;

    // Exploitation: reward successful operators
    double insertionRate = (insertions + 1) / (count + 1);
    double improvementRate = (improvements + 1) / (count + 1);

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
