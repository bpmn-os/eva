#pragma once

#include "../eva.h"

#include <concepts>
#include <type_traits>

namespace EVA {

template < typename Individual, typename Genome = Individual >
requires (
  std::is_constructible_v<Individual, const Genome&>
)
std::shared_ptr< const Individual > constructor( [[maybe_unused]] const EvolutionaryAlgorithm<Individual,Genome>* eva, const Genome& genome ) {
  // Use constructor
  return std::make_shared<const Individual>(genome);
}

} // end namespace EVA

