#pragma once

#include "../eva.h"

#include <concepts>
#include <type_traits>

namespace EVA {

template < typename Individual, typename Genome = Individual >
requires (
  requires(const Individual& i) {
    { i.getFitness() } -> std::same_as<Fitness>;
  }
)
Fitness fitnessFunction( [[maybe_unused]] const EvolutionaryAlgorithm<Individual,Genome>* eva, const std::shared_ptr< const Individual >& individual ) {
  return individual->getFitness();
}

} // end namespace EVA

