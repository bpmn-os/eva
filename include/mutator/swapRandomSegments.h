#pragma once

#include "../eva.h"

#include <concepts>
#include <type_traits>
#include <ranges>
#include <algorithm>

namespace EVA {

template < typename Individual, typename Genome = Individual >
requires (
  std::ranges::range<Genome> &&
  (
    std::ranges::random_access_range<Genome> || 
    requires(Genome genome, typename Genome::value_type value) { { genome.push_back(value) }; } 
  )
)
std::function<Genome(EvolutionaryAlgorithm<Individual,Genome>*, const std::vector< std::shared_ptr< const Individual > >&)> swapRandomSegments() {
  return []( EvolutionaryAlgorithm<Individual,Genome>* eva, const std::vector< std::shared_ptr< const Individual > >& individuals ) {
    const Genome& genome = *individuals.begin()->get();
    auto size = genome.size();

    if (size < 2) {
      throw std::logic_error("swapRandomSegments: genome must have a length of at least 2");
    }

    std::set<size_t> sortedIndices;
    while ( sortedIndices.size() < 4 ) {
      sortedIndices.insert( eva->randomIndex( size - 1) );
    }

    // Get the 4 sorted indices
    auto it1 = sortedIndices.begin();
    auto i1 = *it1++;
    auto j1 = *it1++;
    auto i2 = 1 + *it1++; // increase by 1 to ensure non-overlapping segments
    auto j2 = 1 + *it1++;
    
    
    auto first1 = std::next(genome.begin(), i1);  
    auto last1 = std::next(first1, j1-i1 );                    
    auto first2 = std::next(last1, i2-j1);              
    auto last2 = std::next(first2, j2-i2);      
     
    // Build mutant by inserting segments in swapped order
    Genome mutant;
    if constexpr (requires { mutant.reserve(genome.size()); }) {
      mutant.reserve(genome.size());
    }
    
    // Copy initial segment
    for (auto it = genome.begin(); it != first1; ++it) {
      mutant.push_back(*it);
    }

    // Copy segment 2
    for (auto it = first2; it != std::next(last2); ++it) {
      mutant.push_back(*it);
    }

    // Copy segment between 1 and 2
    for (auto it = std::next(last1); it != first2; ++it) {
      mutant.push_back(*it);
    }

    // Copy segment 1
    for (auto it = first1; it != std::next(last1); ++it) {
      mutant.push_back(*it);
    }

    // Copy final segment
    for (auto it = std::next(last2); it != genome.end(); ++it) {
      mutant.push_back(*it);
    }

    return mutant;
  };
}

} // end namespace EVA

