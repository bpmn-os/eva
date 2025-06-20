#pragma once

#include "../eva.h"

#include <concepts>
#include <type_traits>

namespace EVA {

template < typename Individual, typename Genome = Individual >
std::function<std::shared_ptr< const Individual >(EvolutionaryAlgorithm<Individual,Genome>*)> randomSelection() {
  return []( EvolutionaryAlgorithm<Individual,Genome>* eva ) {
    auto& population = eva->getPopulation();
    auto index = eva->randomIndex( population.size() );

    return population[index].first;
  };
}

} // end namespace EVA

