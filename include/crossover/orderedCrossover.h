#pragma once

#include "../eva.h"

#include <concepts>
#include <type_traits>
#include <unordered_set>
#include <algorithm>
#include <ranges>

namespace EVA {

template < typename Individual, typename Genome = Individual >
requires (
  std::ranges::range<Genome> &&
  (
    std::ranges::random_access_range<Genome> || 
    requires(Genome genome, typename Genome::value_type value) { { genome.push_back(value) }; } 
  )
)
std::function<Genome(const EvolutionaryAlgorithm<Individual,Genome>*, const Genome&, const Genome&)> orderedCrossover() {
  return []( const EvolutionaryAlgorithm<Individual,Genome>* eva, const Genome& parent1, const Genome& parent2 ) {
    auto size = parent1.size();

    if ( parent2.size() != size ) {
      throw std::logic_error("orderedCrossover: parents must have the same length");
    }
    if ( size < 2 ) {
      throw std::logic_error("orderedCrossover: parents must have a length > 1");
    }

    // Randomly determine two positions
    size_t i = eva->randomIndex( size );
    size_t j = eva->randomIndex( size );
    while (j == i) {
      j = eva->randomIndex( size );
    }

    if (i > j) {
      std::swap(i, j);
    }
  
    if constexpr (std::ranges::random_access_range<Genome>) {
      Genome offspring(size);
      std::unordered_set<typename Genome::value_type> inserted;

      // Copy segment from parent1
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
    else {
      Genome segment;
      std::unordered_set<typename Genome::value_type> inserted;

      // Copy segment from parent1
      auto it1 = std::ranges::next(parent1.begin(), i);
      for (size_t k = i; k <= j; ++k) {
        segment.push_back( *it1 );
        inserted.insert( *it1 );
        ++it1;
      }
      
      Genome offspring;
      auto it2 = parent2.begin();

      // Add elements from parent2 before i, skipping those in segment
      for (size_t k = 0; k < i; ++k) {
        while ( inserted.contains(*it2) ) {
          // offspring contains element at index, continue with next index
          ++it2;
        }
    
        offspring.push_back(*it2);
        ++it2;
      }      

      // Append the copied segment
      offspring.insert(offspring.end(), segment.begin(), segment.end());
     
      // Add elements from parent2 after j, skipping those in segment
      for (size_t k = j+1; k < size; ++k) {
        while ( inserted.contains(*it2) ) {
          // offspring contains element at index, continue with next index
          ++it2;
        }
    
        offspring.push_back(*it2);
        ++it2;
      }      

      return offspring;      
    }
  };
}

} // end namespace EVA

