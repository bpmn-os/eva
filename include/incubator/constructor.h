#pragma once

#include "../eva.h"

#include <concepts>
#include <type_traits>

namespace EVA {

template < typename Individual, typename Genome = Individual >
requires (
  std::is_constructible_v<Individual, const Genome&>
)
std::function<std::shared_ptr<const Individual>(const EvolutionaryAlgorithm<Individual, Genome>*, const Genome&)>
constructor() {
  return []( [[maybe_unused]] const EvolutionaryAlgorithm<Individual,Genome>* eva, const Genome& genome ) {
    // Use constructor
    return std::make_shared<const Individual>(genome);
  };
}

} // end namespace EVA

