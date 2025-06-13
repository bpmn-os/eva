#pragma once

#include "../eva.h"
#include <random>
#include <numeric>

namespace EVA {

template <typename Individual, typename Genome = Individual>
requires (
  std::ranges::random_access_range<Genome> &&
  std::is_integral_v<typename Genome::value_type>
)
std::function<std::pair< std::shared_ptr< const Individual >, Fitness >(const EvolutionaryAlgorithm<Individual, Genome>*)> 
randomPermutation(unsigned int length) {
  return [length](const EvolutionaryAlgorithm<Individual, Genome>* eva) {
    Genome genome(length);
    std::iota(genome.begin(), genome.end(), 1);
    std::mt19937 g(std::random_device{}());
    std::shuffle(genome.begin(), genome.end(), g);
    auto threadConfig = eva->getThreadConfig();
    auto individual = threadConfig->incubate(eva, genome);
    auto fitness = threadConfig->evaluate(eva, individual);
    return std::make_pair(individual, fitness);
  };
}

} // namespace EVA

