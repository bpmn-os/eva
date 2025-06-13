#pragma once

#include <vector>
#include <set>
#include <utility>
#include <concepts>
#include <mutex>
#include <thread>
#include <random>
#include <memory>
#include <functional>

namespace EVA {

using Fitness = std::vector<double>;

template < typename Individual, typename Genome = Individual >
requires (
  std::movable<Individual> && 
  std::movable<Genome> && 
  std::is_convertible_v<Individual,Genome>
)
class EvolutionaryAlgorithm {
public:
  struct Config {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;
    unsigned int seed = std::random_device{}();
    unsigned int threads = std::max(1u, std::thread::hardware_concurrency());
    size_t minPopulationSize = 2;
    size_t maxPopulationSize = 100;
    double crossoverProbability = 0.8;
    double mutationProbability = 0.2;
    std::function<std::shared_ptr< const Individual >(const EVA*)> parentSelection = nullptr;
    std::function<std::shared_ptr< const Individual >(const EVA*)> alternativeParentSelection = nullptr;
    std::function<Genome(const EVA*, const Genome&, const Genome&)> crossover = nullptr;
    std::function<std::shared_ptr< const Individual >(const EVA*)> mutationSelection = nullptr;
    std::function<Genome(const EVA*, const Genome&)> mutate = nullptr;
    std::function<std::shared_ptr< const Individual >(const EVA*, const Genome&)> incubate = nullptr;
    std::function<Fitness(const EVA*, const std::shared_ptr< const Individual >&)> evaluate = nullptr;
    std::function<bool(const EVA*)> termination = nullptr;
    std::function<void(const EVA*, const std::shared_ptr< const Individual >&, const Fitness&)> monitor = nullptr;
  };
  static Config default_config() { return {}; } // Work around for compiler bug see: https://stackoverflow.com/questions/53408962/try-to-understand-compiler-error-message-default-member-initializer-required-be/75691051#75691051

  struct Comparator {
    const std::vector< std::pair< std::shared_ptr< const Individual >, Fitness > >* populationPtr;

    Comparator(const std::vector< std::pair< std::shared_ptr< const Individual >, Fitness > >* populationPtr)
      : populationPtr(populationPtr)
    {
    };

    bool operator()(size_t lhs, size_t rhs) const {
      const Fitness& lhsFitness = (*populationPtr)[lhs].second;
      const Fitness& rhsFitness = (*populationPtr)[rhs].second;
      return lhsFitness >= rhsFitness; // std::vector<double> supports lexicographical comparison via operator>=
    }
  };

  EvolutionaryAlgorithm(Config config = default_config())
    : config(config)
    , orderedIndices(Comparator(&population))
  {
    if ( config.parentSelection && !config.crossover ) {
      throw std::logic_error("EvolutionaryAlgorithm: crossover operator missing");
    }
    if ( !config.parentSelection && config.crossover ) {
      throw std::logic_error("EvolutionaryAlgorithm: parent selector missing");
    }
    if ( !config.alternativeParentSelection ) {
      config.alternativeParentSelection = config.parentSelection;
    }
    if ( config.mutationSelection && !config.mutate ) {
      throw std::logic_error("EvolutionaryAlgorithm: mutator missing");
    }
    if ( !config.mutationSelection && config.mutate ) {
      throw std::logic_error("EvolutionaryAlgorithm: mutation selector missing");
    }
    if ( !config.crossover && !config.mutate ) {
      throw std::logic_error("EvolutionaryAlgorithm: crossover or mutator needed");
    }
    if ( !config.crossover ) {
      config.crossoverProbability = 0.0;
      config.mutationProbability = 1.0;
    }
    if ( !config.mutate ) {
      config.crossoverProbability = 1.0;
      config.mutationProbability = 0.0;
    }    
    if ( std::abs(config.crossoverProbability + config.mutationProbability - 1.0) > 1e-10 ) {
      throw std::logic_error("EvolutionaryAlgorithm: crossover and mutation probabilities must add up to 1");
    }
    
    if ( !config.incubate ) {
      throw std::logic_error("EvolutionaryAlgorithm: incubator missing");
    }
    if ( !config.evaluate ) {
      throw std::logic_error("EvolutionaryAlgorithm: evaluator missing");
    }
    if ( !config.termination ) {
      throw std::logic_error("EvolutionaryAlgorithm: termination condition missing");
    }
    if ( config.minPopulationSize < 1  ) {
      throw std::logic_error("EvolutionaryAlgorithm: minimum population size must be at least 1");
    }

  };
  
  /// Adds an evaluated individual to the population without exceeding the maximum population size
  void add( std::shared_ptr< const Individual > individual, Fitness fitness ) {
    auto lock = acquireLock();
    
    if ( config.monitor ) {
      // allows to inspect added individual before it is inserted
      // a lock on the population has already been acquired
      // population still contains the individual that may be replaced
      // use getWorst(true) to access this individual
      config.monitor( this, individual, fitness );
    }
    
    if ( population.size() < config.maxPopulationSize ) {
      // add individual to population
      size_t index = population.size();
      population.emplace_back(std::move(individual), std::move(fitness));
      orderedIndices.insert(index);
    }
    else {
      // replace worst individual in population
      size_t index = *orderedIndices.rbegin();
      population[index] = std::make_pair(std::move(individual), std::move(fitness));
      orderedIndices.erase(std::prev(orderedIndices.end()));
      orderedIndices.insert(index);
    }
  }
  
  /// Returns a random index between 0 and size - 1
  [[nodiscard]] size_t randomIndex( size_t size ) const {
    return std::uniform_int_distribution<size_t>(0, size - 1)(randomNumberGenerator);
  }

  /// Returns a random probability between 0 and 1
  [[nodiscard]] double randomProbability() const {
    return std::uniform_real_distribution<double>(0, 1)(randomNumberGenerator);
  }
  
  /// Returns population of individuals with their hierarchically ordered 
  [[nodiscard]] const std::vector< std::pair< std::shared_ptr< const Individual >, Fitness > >& getPopulation() const {  
    return population;
  }
  
  /// Returns fitness ordered set of population indices
  [[nodiscard]] const std::set<size_t, Comparator>& getOrderedIndices() const {
    return orderedIndices;
  }

  void run() {
    if ( population.size() < config.minPopulationSize ) {
      throw std::logic_error("EvolutionaryAlgorithm: population too small");
    }
  
    terminate = false;
    std::vector<std::jthread> workers;

    for (unsigned int index = 1; index <= config.threads; ++index) {
      workers.emplace_back(
        [this,index](std::stop_token) {
          runThread(index);
        }
      );
    }
  }

  /// Returns a lock guard for thread-safe access to the population
  [[nodiscard]] std::lock_guard<std::mutex> acquireLock() const {
    return std::lock_guard<std::mutex>(populationMutex);
  }  
  
  std::pair<std::shared_ptr<const Individual>, Fitness> getBest(bool locked = false) const {
    std::unique_lock<std::mutex> lock;
    if (!locked) {
      lock = std::unique_lock(populationMutex);
    }
    if (population.empty()) {
      return { nullptr, {} };
    }
    size_t index = *orderedIndices.begin(); // get index of best
    return population[index];
  }

  std::pair<std::shared_ptr<const Individual>, Fitness> getWorst(bool locked = false) const {
    std::unique_lock<std::mutex> lock;
    if (!locked) {
      lock = std::unique_lock(populationMutex);
    }
    if (population.empty()) {
      return { nullptr, {} };
    }
    size_t index = *orderedIndices.rbegin(); // get index of worst
    return population[index];
  }
  
  const Config& getConfig() const { return config; }
  static size_t getThreadIndex() { return threadIndex; };
protected:
  Config config;
  mutable std::mutex populationMutex;
  std::vector< std::pair< std::shared_ptr< const Individual >, Fitness> > population; ///< Population of individuals with their hierarchically ordered fitness
  std::set<size_t, Comparator> orderedIndices; ///< Fitness ordered set of population indices
  static thread_local std::mt19937 randomNumberGenerator;
  static thread_local size_t threadIndex;
  std::atomic<bool> terminate;


  void runThread(unsigned int index) {
    randomNumberGenerator.seed( config.seed + index );
    threadIndex = index;
    do {
      if ( population.size() > 1 && randomProbability() < config.crossoverProbability   ) {
        auto parent1 = config.parentSelection( this );
        auto parent2 = config.alternativeParentSelection( this );
        while ( parent1.get() == parent2.get() ) {
          parent2 = config.alternativeParentSelection( this );
        }
        auto& genome1 = *parent1.get();
        auto& genome2 = *parent2.get();
        auto offspring = config.incubate( this, config.crossover( this, genome1, genome2 ) );
        auto fitness = config.evaluate( this, offspring );
        add( offspring, fitness );
      }
      else {
        auto individual = config.mutationSelection( this );
        auto& genome = *individual.get();
        auto mutant = config.incubate( this, config.mutate( this, genome ) );
        auto fitness = config.evaluate( this, mutant );
        add( mutant, fitness );
      }
      
      if ( config.termination( this ) ) {
        terminate = true;
      }
    } while ( !terminate );
  }
};

template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome>
)
thread_local std::mt19937 EvolutionaryAlgorithm<Individual, Genome>::randomNumberGenerator;

template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome>
)
thread_local size_t EvolutionaryAlgorithm<Individual, Genome>::threadIndex = 0;

} // end namespace EVA

