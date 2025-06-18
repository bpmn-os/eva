#pragma once

#include "../eva.h"

#include <concepts>
#include <type_traits>

namespace EVA {

template < typename Individual, typename Genome = Individual >
std::function<std::shared_ptr< const Individual >(EvolutionaryAlgorithm<Individual,Genome>*)> rankSelection() {
  return []( EvolutionaryAlgorithm<Individual,Genome>* eva ) {
    auto& population = eva->getPopulation();
    auto& orderedIndices = eva->getOrderedIndices();
    const size_t n = population.size();

    // Rank-based selection assigns a 'weight' based on rank.
    // Best individual gets weight `n`, 2nd-best individual gets weight `n-1`, ..., worst individual gets weight `1`.
    double total_weigth = (double)(n * (n + 1)) / 2.0;
    double spin = eva->randomProbability() * total_weigth;
    double sum = 0.0;
    size_t weight = n; // start with the highest weight

    // iterate through the indices in the ordered set (which are ordered by fitness)
    for (size_t index : orderedIndices) {
      sum += weight;
      if ( sum >= spin ) {
        return population[index].first;
      }
      weight--; // reduce weight for next individual
    }
    return population[*orderedIndices.rbegin()].first;
  };
}

} // end namespace EVA
