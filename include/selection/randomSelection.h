#pragma once

#include "../eva.h"

#include <concepts>
#include <type_traits>

namespace EVA {

template < typename Individual, typename Genome = Individual >
std::shared_ptr< const Individual > randomSelection( const EvolutionaryAlgorithm<Individual,Genome>* eva ) {
  auto lock = eva->acquireLock();
  auto& population = eva->getPopulation();
  auto index = eva->randomIndex( population.size() );

  return population[index].first;
}

} // end namespace EVA

