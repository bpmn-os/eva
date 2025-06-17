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
std::function<Genome(const EvolutionaryAlgorithm<Individual,Genome>*, const std::vector< std::shared_ptr< const Individual > >&)> randomSwap() {
  return []( const EvolutionaryAlgorithm<Individual,Genome>* eva, const std::vector< std::shared_ptr< const Individual > >& individuals ) {
    const Genome& genome = *individuals.begin()->get();
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

    if constexpr (std::ranges::random_access_range<Genome>) {
      std::ranges::swap(mutant[i], mutant[j]);
    }
    else {
      auto it1 = std::ranges::next(mutant.begin(), i);
      auto it2 = std::ranges::next(mutant.begin(), j);
      std::ranges::swap(*it1,*it2);
    }

    return mutant;
  };
}

} // end namespace EVA

