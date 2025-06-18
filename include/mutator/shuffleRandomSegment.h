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
std::function<Genome(EvolutionaryAlgorithm<Individual,Genome>*, const std::vector< std::shared_ptr< const Individual > >&)> shuffleRandomSegment() {
  return []( EvolutionaryAlgorithm<Individual,Genome>* eva, const std::vector< std::shared_ptr< const Individual > >& individuals ) {
    const Genome& genome = *individuals.begin()->get();
    auto size = genome.size();

    if (size < 2) {
      throw std::logic_error("shuffleRandomSegment: genome must have a length > 2");
    }

    // Randomly shuffle all element between (and including) two elements in the genome
    auto mutant(genome);
    size_t i = eva->randomIndex( size );
    size_t j = eva->randomIndex( size );
    while (j == i) {
      j = eva->randomIndex( size );
    }
    
    if (i > j) std::swap(i, j);

    std::ranges::shuffle(mutant.begin() + i, mutant.begin() + j + 1, eva->getRandomNumberGenerator());

    return mutant;
  };
}

} // end namespace EVA

