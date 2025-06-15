#pragma once

#include "../eva.h"
#include <random>
#include <numeric>
#include <ranges>

namespace EVA {

template <typename Individual, typename Genome = Individual>
requires (
  std::is_integral_v<typename Genome::value_type> &&
  std::ranges::range<Genome> &&
  (
    std::ranges::random_access_range<Genome> || 
    requires(Genome genome, typename Genome::value_type value) { { genome.push_back(value) }; } 
  )
)
std::function<std::pair< std::shared_ptr< const Individual >, Fitness >(const EvolutionaryAlgorithm<Individual, Genome>*)> 
randomPermutation(size_t length) {
  return [length](const EvolutionaryAlgorithm<Individual, Genome>* eva) {
    std::vector<typename Genome::value_type> permutation(length);
    std::iota(permutation.begin(), permutation.end(), 1);
    std::mt19937 randomGenerator(std::random_device{}());
    std::shuffle(permutation.begin(), permutation.end(), randomGenerator);

    Genome genome;
    if constexpr (std::ranges::random_access_range<Genome>) {
      genome = Genome(permutation.begin(), permutation.end());
    } 
    else {
      for (auto& v : permutation) {
        genome.push_back(v);
      }
    }

    auto threadConfig = eva->getThreadConfig();
    auto individual = threadConfig->incubate(eva, genome);
    auto fitness = threadConfig->evaluate(eva, individual);
    return std::make_pair(individual, fitness);
  };
}

} // namespace EVA

