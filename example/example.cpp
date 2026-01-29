#include "../include/eva.h"
#include "../include/selection/binaryTournamentSelection.h"
#include "../include/selection/rankSelection.h"
#include "../include/selection/randomSelection.h"
#include "../include/crossover/orderedCrossover.h"
#include "../include/mutator/randomSwap.h"
#include "../include/mutator/shuffleRandomSegment.h"
#include "../include/mutator/swapRandomSegments.h"
#include "../include/incubator/constructor.h"
#include "../include/evaluator/fitnessFunction.h"
#include "../include/spawn/randomPermutation.h"
#include "../include/calibration/weightUpdate.h"

#include <iostream>

#include <string>
#include <vector>
#include <random>
#include <ranges>

using Values = std::vector<unsigned int>;

/// Permutation class representing individuals in the population
struct Permutation {
  Values values; /// The genome
  /// Determines a fitness value between 0 and 1 (optimal).
  [[nodiscard]] EVA::Fitness getFitness() const {
    if ( values.empty() ) throw std::logic_error("Cannot evaluate permutation");
    // determine the ratio of elements that are smaller or equal to their successor
    int count = 0;
    for (size_t i = 1; i < values.size(); ++i) {
      if (values[i - 1] <= values[i]) { 
        ++count;
      }
    }
    return { (double)count / ( values.size() - 1 ) };
  };
  
  // Construct from const std::vector<unsigned int>
  Permutation( Values values ) : values(std::move(values)) {};
  // Cast to const std::vector<unsigned int>&
  operator const Values&() const {
    return values;
  }

  // Equality comparison
  bool operator==(const Permutation& other) const {
    // consider using predetermined hash to avoid element-wise comparisons
    return values == other.values;
  }
};

/// Helper function to print out individuals and fitness
template <typename Range>
[[nodiscard]] std::string stringify(const Range& values) {
  if ( values.empty() ) throw std::logic_error("Illegal range");
  std::string result;
  for ( auto value : values ) {
    result += std::string(", ") + std::to_string(value);
  }
  result.front() = '[';
  result += " ]";
  return result;
}

int main(int argc, char** argv) {
  size_t length;
  auto prompt = std::format("Enter the length of the permutation (at least 5): ");
  std::cout << prompt;
  std::cin >> length;
  if (length < 5) {
    length = 5;
  }
  std::cout << std::format("Using permutation length: {}\n", length);
    
  // Create instance of evolutionary algorithm with inline configuration
  EVA::EvolutionaryAlgorithm< Permutation, Values > eva({
    .threads = 8,
    .minPopulationSize = 50,
    .maxPopulationSize = 100,
    .maxComputationTime = 60,
//    .maxSolutionCount = 100000,
    .maxNonImprovingSolutionCount = 100000,
    .threadConfig = {
      // use default implementations
      .spawn = EVA::randomPermutation<Permutation, Values>(length),
      .reproduction = {
        // Crossover with different selectors
        { {EVA::binaryTournamentSelection<Permutation, Values>(), EVA::binaryTournamentSelection<Permutation, Values>()}, EVA::orderedCrossover<Permutation, Values>(), 1.0 },
        { {EVA::binaryTournamentSelection<Permutation, Values>(), EVA::rankSelection<Permutation, Values>()}, EVA::orderedCrossover<Permutation, Values>(), 1.0 },
        { {EVA::binaryTournamentSelection<Permutation, Values>(), EVA::randomSelection<Permutation, Values>()}, EVA::orderedCrossover<Permutation, Values>(), 1.0 },
        { {EVA::rankSelection<Permutation, Values>(), EVA::rankSelection<Permutation, Values>()}, EVA::orderedCrossover<Permutation, Values>(), 1.0 },
        { {EVA::rankSelection<Permutation, Values>(), EVA::randomSelection<Permutation, Values>()}, EVA::orderedCrossover<Permutation, Values>(), 1.0 },
        { {EVA::randomSelection<Permutation, Values>(), EVA::randomSelection<Permutation, Values>()}, EVA::orderedCrossover<Permutation, Values>(), 1.0 },
        // Mutations with different selection strategies
        { {EVA::binaryTournamentSelection<Permutation, Values>()}, EVA::randomSwap<Permutation, Values>(), 0.1 },
        { {EVA::rankSelection<Permutation, Values>()}, EVA::randomSwap<Permutation, Values>(), 0.1 },
        { {EVA::randomSelection<Permutation, Values>()}, EVA::randomSwap<Permutation, Values>(), 0.1 },
        { {EVA::binaryTournamentSelection<Permutation, Values>()}, EVA::shuffleRandomSegment<Permutation, Values>(), 0.1 },
        { {EVA::rankSelection<Permutation, Values>()}, EVA::shuffleRandomSegment<Permutation, Values>(), 0.1 },
        { {EVA::randomSelection<Permutation, Values>()}, EVA::shuffleRandomSegment<Permutation, Values>(), 0.1 },
        { {EVA::binaryTournamentSelection<Permutation, Values>()}, EVA::swapRandomSegments<Permutation, Values>(), 0.1 },
        { {EVA::rankSelection<Permutation, Values>()}, EVA::swapRandomSegments<Permutation, Values>(), 0.1 },
        { {EVA::randomSelection<Permutation, Values>()}, EVA::swapRandomSegments<Permutation, Values>(), 0.1 }
      },
      .calibration = EVA::weightUpdate<Permutation, Values>(0.1)  // Use adaptive weight update with learning rate 0.1
    },
    .incubate = EVA::constructor<Permutation, Values>(),
    .evaluate = EVA::fitnessFunction<Permutation, Values>(),
    // create lambda
    .termination = []( [[maybe_unused]] EVA::EvolutionaryAlgorithm< Permutation, Values >* eva) {
      auto [bestPermutation, bestFitness] = eva->getBest();
      // return true if best permutation is perfectly ordered
      return ( bestFitness[0] > 1 - 1e-10 );
    },
    // create lambda
    .monitor = []( [[maybe_unused]] EVA::EvolutionaryAlgorithm< Permutation, Values >* eva, const std::shared_ptr< const Permutation >& permutation, const EVA::Fitness& fitness) {
      std::cout << eva->getSolutionCount() << ". ";
      std::cout << "(" << eva->getNonImprovingSolutionCount() << ") ";
      auto [bestPermutation, bestFitness] = eva->getBest(true);
      if ( bestPermutation ) {
        std::cout << "Previous best: ( " << stringify(bestPermutation->values) << ", " << stringify(bestFitness) << " ) - ";
      }
      std::cout << "New: ( " << stringify(permutation->values) << ", " << stringify(fitness) << " )\n";
    }
  });
  
  // run evolutionary algorithm with given number of threads
  eva.run();
  
  // obtain best solution found
  auto [bestPermutation, bestFitness] = eva.getBest();
  std::cout << "\nBest: " << stringify(bestPermutation->values) << ", fitness: " << stringify(bestFitness) << "\n";
}
