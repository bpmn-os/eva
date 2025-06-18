#pragma once

#include "../eva.h"

#include <concepts>
#include <type_traits>

namespace EVA {

template <typename Individual, typename Genome = Individual>
  requires (std::is_convertible_v<Individual, Genome>)
std::function<std::shared_ptr< const Individual >(EvolutionaryAlgorithm<Individual,Genome>*)> binaryTournamentSelection() {
  return [](EvolutionaryAlgorithm<Individual, Genome>* eva) {
    const auto& population = eva->getPopulation();
    size_t size = population.size();

    if (size < 2) throw std::logic_error("Binary tournament requires at least 2 individuals");

    size_t i = eva->randomIndex(size);
    size_t j = eva->randomIndex(size);
    while (j == i) j = eva->randomIndex(size); // ensure two distinct individuals

    const auto& candidate1 = population[i];
    const auto& candidate2 = population[j];

    const auto& winner = (candidate1.second >= candidate2.second) ? candidate1.first : candidate2.first;

    return winner;
  };
}


} // end namespace EVA

