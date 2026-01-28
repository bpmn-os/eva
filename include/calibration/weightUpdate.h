#pragma once

#include "../eva.h"

namespace EVA {

/**
 * @brief Default adaptive weight update strategy
 *
 * Increases the weight of successful strategies (those producing new best solutions)
 * by the adaptation rate, while decreasing all other weights proportionally.
 *
 * @tparam Individual The individual type
 * @tparam Genome The genome type
 */
template <typename Individual, typename Genome = Individual>
std::function<void(EvolutionaryAlgorithm<Individual, Genome>*, const std::shared_ptr<const Individual>&, size_t, const Fitness&, bool, bool)>
weightUpdate() {
  return [](EvolutionaryAlgorithm<Individual, Genome>* eva, const std::shared_ptr<const Individual>& offspring, size_t reproducer, const Fitness& fitness, bool isDuplicate, bool isFittest) {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;
    auto threadConfig = eva->getThreadConfig();

    (void)offspring;  // Not used in default implementation
    (void)fitness;  // Not used in default implementation
    (void)isDuplicate;  // Not used in default implementation

    // Only update if this offspring was the new fittest
    if (isFittest) {
      // Scale down all weights and increase the successful one
      for (auto& weight : EVA::weights) {
        weight -= threadConfig->adaptationRate * weight;
      }
      EVA::weights[reproducer] += threadConfig->adaptationRate * EVA::totalWeight;
    }
  };
}

} // namespace EVA
