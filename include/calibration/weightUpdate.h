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
 * @param adaptationRate Learning rate (0.0 = no learning, 1.0 = instant). Default 0.1.
 *        Higher values adapt faster but may be less stable.
 */
template <typename Individual, typename Genome = Individual>
std::function<void(EvolutionaryAlgorithm<Individual, Genome>*, const std::shared_ptr<const Individual>&, size_t, const Fitness&, bool, bool)>
weightUpdate(double adaptationRate = 0.1) {
  return [adaptationRate]([[maybe_unused]] EvolutionaryAlgorithm<Individual, Genome>* eva, [[maybe_unused]] const std::shared_ptr<const Individual>& offspring, size_t reproducer, [[maybe_unused]] const Fitness& fitness, [[maybe_unused]] bool isDuplicate, bool isFittest) {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;

    // Only update if this offspring was the new fittest
    if (isFittest) {
      // Scale down all weights and increase the successful one
      for (auto& weight : EVA::weights) {
        weight -= adaptationRate * weight;
      }
      EVA::weights[reproducer] += adaptationRate * EVA::totalWeight;
    }
  };
}

} // namespace EVA
