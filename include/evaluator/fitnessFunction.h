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
std::function<Fitness(EvolutionaryAlgorithm<Individual,Genome>*, const std::shared_ptr< const Individual >&)> fitnessFunction() {
  return []( [[maybe_unused]] EvolutionaryAlgorithm<Individual,Genome>* eva, const std::shared_ptr< const Individual >& individual ) {
    return individual->getFitness();
  };
}

} // end namespace EVA

