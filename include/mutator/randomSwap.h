#pragma once

#include "../eva.h"

#include <concepts>
#include <type_traits>
#include <ranges>
#include <algorithm>

namespace EVA {

template < typename Individual, typename Genome = Individual >
requires (
  std::ranges::range<Genome>
)
std::function<Genome(const EvolutionaryAlgorithm<Individual,Genome>*, const Genome&)> randomSwap() {
  return []( const EvolutionaryAlgorithm<Individual,Genome>* eva, const Genome& genome ) {
    auto size = genome.size();

    if (size < 2) {
      throw std::logic_error("randomSwap: genome must have a length > 2");
    }

    // Randomly swap two elements in the genome
    auto mutant(genome);
    size_t i = eva->randomIndex( size );
    size_t j = eva->randomIndex( size );
    while (j == i) {
      j = eva->randomIndex( size );
    }

    std::ranges::swap(mutant[i], mutant[j]);

    return mutant;
  };
}

} // end namespace EVA

