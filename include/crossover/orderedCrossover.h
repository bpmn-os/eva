#pragma once

#include "../eva.h"

#include <concepts>
#include <type_traits>
#include <unordered_set>
#include <algorithm>

namespace EVA {

template < typename Individual, typename Genome = Individual >
requires (
  std::ranges::range<Genome>
)
Genome orderedCrossover( const EvolutionaryAlgorithm<Individual,Genome>* eva, const Genome& parent1, const Genome& parent2 ) {
  if ( parent2.size() != parent1.size() ) {
    throw std::logic_error("orderedCrossover: parents must have the same length");
  }
  if ( parent1.size() < 2 ) {
    throw std::logic_error("orderedCrossover: parents must have a length > 1");
  }

  // Randomly determine two positions
  auto size = parent1.size();
  size_t i = eva->randomIndex( size );
  size_t j = eva->randomIndex( size );
  while (j == i) {
    j = eva->randomIndex( size );
  }

  if (i > j) {
    std::swap(i, j);
  }
  
  Genome offspring(size);
  std::unordered_set<typename Genome::value_type> inserted;

  // Copy slice from parent1
  for (size_t k = i; k <= j; ++k) {
    offspring[k] = parent1[k];
    inserted.insert(parent1[k]);
  }

  // Fill rest from parent2
  size_t p2_index = 0;
  for (size_t k = 0; k < size; ++k) {
    if (k >= i && k <= j) {
      // index belong to slice of parent1
      continue;
    }
    
    while ( inserted.contains(parent2[p2_index]) ) {
      // offspring contains element at index, continue with next index
      ++p2_index;
    }
    
    offspring[k] = parent2[p2_index++];
  }

  return offspring;
}

} // end namespace EVA

