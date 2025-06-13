#include "../include/eva.h"
#include "../include/selection/binaryTournamentSelection.h"
#include "../include/selection/rankSelection.h"
#include "../include/selection/randomSelection.h"
#include "../include/crossover/orderedCrossover.h"
#include "../include/mutator/randomSwap.h"
#include "../include/incubator/constructor.h"
#include "../include/evaluator/fitnessFunction.h"

#include <iostream>

#include <string>
#include <vector>
#include <random>
#include <ranges>

/// Permutation class representing individuals in the population
struct Permutation {
  std::vector<unsigned int> values; /// The genome
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
  Permutation( std::vector<unsigned int> values ) : values(std::move(values)) {};
  // Cast to const std::vector<unsigned int>&
  operator const std::vector<unsigned int>&() const {
    return values;
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
  auto prompt = std::format("Enter the length of the permutation: ");
  std::cout << prompt;
  std::cin >> length;
  std::cout << std::format("Using permutation length: {}\n", length);
  
  std::atomic<unsigned int> counter = 0;
  unsigned int maxIterations = 100;
  
  // Create instance of evolutionary algorithm with inline configuration
  EVA::EvolutionaryAlgorithm< Permutation, std::vector<unsigned int> > eva({
    .minPopulationSize = 5,
    .maxPopulationSize = 20,
    // use default implementations
    .parentSelection = &EVA::rankSelection<Permutation, std::vector<unsigned int>>,
    .alternativeParentSelection = &EVA::binaryTournamentSelection<Permutation, std::vector<unsigned int>>,
    .crossover = &EVA::orderedCrossover<Permutation, std::vector<unsigned int>>,
    .mutationSelection = &EVA::randomSelection<Permutation, std::vector<unsigned int>>,
    .mutate = &EVA::randomSwap<Permutation, std::vector<unsigned int>>,
    .incubate = &EVA::constructor<Permutation, std::vector<unsigned int>>,
//    .evaluate = &EVA::fitnessFunction<Permutation, std::vector<unsigned int>>
    // create lambdas
    .evaluate = []( [[maybe_unused]] const EVA::EvolutionaryAlgorithm< Permutation, std::vector<unsigned int> >* eva, const std::shared_ptr< const Permutation >& permutation ) { 
      return permutation->getFitness(); 
    },
    // create lambdas using captures
    .termination = [&counter,maxIterations]( [[maybe_unused]] const EVA::EvolutionaryAlgorithm< Permutation, std::vector<unsigned int> >* eva) {
      auto [bestPermutation, bestFitness] = eva->getBest();
      return ( bestFitness[0] > 1 - 1e-10 || counter >= maxIterations ); 
    },
    // create lambda using captures
    .monitor = [&counter]( [[maybe_unused]] const EVA::EvolutionaryAlgorithm< Permutation, std::vector<unsigned int> >* eva, const std::shared_ptr< const Permutation >& permutation, const EVA::Fitness& fitness) {
      counter++;
      std::cout << counter << ". ";       
      auto [bestPermutation, bestFitness] = eva->getBest(true);
      if ( bestPermutation ) {
        std::cout << "Previous best: ( " << stringify(bestPermutation->values) << ", " << stringify(bestFitness) << " ) - ";
      }
      std::cout << "New: ( " << stringify(permutation->values) << ", " << stringify(fitness) << " ) - ";
      std::cout << "Thread: " << eva->getThreadIndex() << "\n";       
    }
  });
  
  // create initial population
  auto config = eva.getConfig();
  for ( unsigned int i = 0; i < config.minPopulationSize; i++ ) {
    // create random genome
    std::vector<unsigned int> genome(length);
    std::iota(genome.begin(), genome.end(), 1);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(genome.begin(), genome.end(), g);
    
    // incubate individual and determine fitness
    auto individual = config.incubate( &eva, genome );
    auto fitness = config.evaluate( &eva, individual );
    
    // add individual
    eva.add( individual, fitness );
  }
  
  // run evolutionary algorithm with given number of threads
  eva.run();
  
  // obtain best solution found
  auto [bestPermutation, bestFitness] = eva.getBest();
  std::cout << "\nBest: " << stringify(bestPermutation->values) << ", fitness: " << stringify(bestFitness) << "\n";
}
