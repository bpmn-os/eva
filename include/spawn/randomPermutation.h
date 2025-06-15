#pragma once

#include "../eva.h"
#include <random>
#include <numeric>

namespace EVA {

template<typename T, typename V>
concept has_push_back = requires(T t, V v) {
  { t.push_back(v) };
};

template <typename Individual, typename Genome = Individual>
requires (
  std::is_integral_v<typename Genome::value_type> &&
  std::ranges::range<Genome> &&
  (std::ranges::random_access_range<Genome> || has_push_back<Genome, typename Genome::value_type>)
)
std::function<std::pair< std::shared_ptr< const Individual >, Fitness >(const EvolutionaryAlgorithm<Individual, Genome>*)> 
randomPermutation(unsigned int length) {
  return [length](const EvolutionaryAlgorithm<Individual, Genome>* eva) {
    std::vector<typename Genome::value_type> permutation(length);
    std::iota(permutation.begin(), permutation.end(), 1);
    std::mt19937 randomGenerator(std::random_device{}());
    std::shuffle(permutation.begin(), permutation.end(), randomGenerator);

    Genome genome;
    if constexpr (std::ranges::random_access_range<Genome>) {
      genome = Genome(permutation.begin(), permutation.end());
    } 
    else /*if (has_push_back<Genome, typename Genome::value_type>)*/ {
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

