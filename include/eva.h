#pragma once

#include <vector>
#include <set>
#include <utility>
#include <concepts>
#include <mutex>
#include <shared_mutex>
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
  struct ThreadConfig {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;
    double crossoverProbability = 0.8;
    double mutationProbability = 0.2;
    std::function<std::pair< std::shared_ptr< const Individual >, Fitness >(const EVA*)> spawn = nullptr;
    std::function<std::shared_ptr< const Individual >(const EVA*)> parentSelection = nullptr;
    std::function<std::shared_ptr< const Individual >(const EVA*)> alternativeParentSelection = nullptr;
    std::function<Genome(const EVA*, const Genome&, const Genome&)> crossover = nullptr;
    std::function<std::shared_ptr< const Individual >(const EVA*)> mutationSelection = nullptr;
    std::function<Genome(const EVA*, const Genome&)> mutate = nullptr;
    std::function<std::shared_ptr< const Individual >(const EVA*, const Genome&)> incubate = nullptr;
    std::function<Fitness(const EVA*, const std::shared_ptr< const Individual >&)> evaluate = nullptr;
  };

  struct Config {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;
    const unsigned int seed = std::random_device{}();
    const unsigned int threads = std::max(1u, std::thread::hardware_concurrency());
    size_t minPopulationSize = 10;
    size_t maxPopulationSize = 100;
    ThreadConfig threadConfig = {};
    std::function<bool(const EVA*)> termination = nullptr;
    std::function<void(const EVA*, const std::shared_ptr< const Individual >&, const Fitness&)> monitor = nullptr;
  };

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

  EvolutionaryAlgorithm(Config config)
    : orderedIndices(Comparator(&population))
  {
    if ( config.minPopulationSize < 2 ) {
      throw std::logic_error("EvolutionaryAlgorithm: minimal population size must be at least 2");
    }
    if ( config.maxPopulationSize < config.minPopulationSize ) {
      throw std::logic_error("EvolutionaryAlgorithm: maximal population size must be at least minimal population");
    }
    if ( config.threadConfig.parentSelection && !config.threadConfig.crossover ) {
      throw std::logic_error("EvolutionaryAlgorithm: crossover operator missing");
    }
    if ( !config.threadConfig.parentSelection && config.threadConfig.crossover ) {
      throw std::logic_error("EvolutionaryAlgorithm: parent selector missing");
    }
    if ( !config.threadConfig.alternativeParentSelection ) {
      config.threadConfig.alternativeParentSelection = config.threadConfig.parentSelection;
    }
    if ( config.threadConfig.mutationSelection && !config.threadConfig.mutate ) {
      throw std::logic_error("EvolutionaryAlgorithm: mutator missing");
    }
    if ( !config.threadConfig.mutationSelection && config.threadConfig.mutate ) {
      throw std::logic_error("EvolutionaryAlgorithm: mutation selector missing");
    }
    if ( !config.threadConfig.crossover && !config.threadConfig.mutate ) {
      throw std::logic_error("EvolutionaryAlgorithm: crossover or mutator needed");
    }
    if ( !config.threadConfig.crossover ) {
      config.threadConfig.crossoverProbability = 0.0;
      config.threadConfig.mutationProbability = 1.0;
    }
    if ( !config.threadConfig.mutate ) {
      config.threadConfig.crossoverProbability = 1.0;
      config.threadConfig.mutationProbability = 0.0;
    }    
    if ( std::abs(config.threadConfig.crossoverProbability + config.threadConfig.mutationProbability - 1.0) > 1e-10 ) {
      throw std::logic_error("EvolutionaryAlgorithm: crossover and mutation probabilities must add up to 1");
    }
    
    if ( !config.threadConfig.incubate ) {
      throw std::logic_error("EvolutionaryAlgorithm: incubator missing");
    }
    if ( !config.threadConfig.evaluate ) {
      throw std::logic_error("EvolutionaryAlgorithm: evaluator missing");
    }
    if ( !config.termination ) {
      throw std::logic_error("EvolutionaryAlgorithm: termination condition missing");
    }
    if ( config.minPopulationSize < 1  ) {
      throw std::logic_error("EvolutionaryAlgorithm: minimum population size must be at least 1");
    }
    
    threadConfigMutex.resize(config.threads);
    threadConfigs.resize(config.threads);
    for (size_t i = 0; i < config.threads; ++i) {
      threadConfigMutex[i] = std::make_unique< std::shared_mutex >();
      threadConfigs[i] = std::make_shared<ThreadConfig>(config.threadConfig);
    }

    globalConfig = std::make_shared<Config>(std::move(config));
    
  };
  
  /// Adds an evaluated individual to the population without exceeding the maximum population size
  void add( std::shared_ptr< const Individual > individual, Fitness fitness ) {
    auto lock = acquireLock();
    
    auto config = getConfig();
    if ( config->monitor ) {
      // allows to inspect added individual before it is inserted
      // a lock on the population has already been acquired
      // population still contains the individual that may be replaced
      // use getWorst(true) to access this individual
      config->monitor( this, individual, fitness );
    }
    
    if ( population.size() < config->maxPopulationSize ) {
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
    auto config = getConfig(); 
    terminate = false;
    std::vector<std::jthread> workers;

    for (unsigned int index = 1; index <= config->threads; ++index) {
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
  
  std::shared_ptr<Config> getConfig() const { 
    std::shared_lock lock(globalConfigMutex);
    return globalConfig; 
  }
  void setConfig(Config config) { 
    std::unique_lock lock(globalConfigMutex);
    globalConfig = std::make_shared<Config>(std::move(config));
  }

  std::shared_ptr<ThreadConfig> getThreadConfig(size_t index = getThreadIndex()) const { 
    if ( index > 0 ) {
      std::shared_lock lock(*threadConfigMutex[index-1]);
      return threadConfigs[index-1];
    }
    else {
      std::shared_lock lock(globalConfigMutex);
      return std::make_shared<ThreadConfig>(globalConfig->threadConfig);
    }
  }

  void setThreadConfig(size_t index, ThreadConfig config) {
    if (index > 0) {
      std::unique_lock lock(*threadConfigMutex[index-1]);
      threadConfigs[index - 1] = std::make_shared<ThreadConfig>(std::move(config));
    }
    else {
      std::unique_lock lock(globalConfigMutex);
      auto global = std::make_shared<Config>(*globalConfig);  // copy
      global->threadConfig = std::move(config);
      globalConfig = std::move(global);
    }
  }
  void setThreadConfig(ThreadConfig config) {
    setThreadConfig(getThreadIndex(), std::move(config));
  }

  static size_t getThreadIndex() { return threadIndex; };
protected:
  std::shared_ptr<Config> globalConfig;
  std::vector< std::shared_ptr<ThreadConfig> > threadConfigs;
  mutable std::mutex populationMutex;
  mutable std::shared_mutex globalConfigMutex;
  mutable std::vector< std::unique_ptr< std::shared_mutex > > threadConfigMutex;
  std::vector< std::pair< std::shared_ptr< const Individual >, Fitness> > population; ///< Population of individuals with their hierarchically ordered fitness
  std::set<size_t, Comparator> orderedIndices; ///< Fitness ordered set of population indices
  static thread_local std::mt19937 randomNumberGenerator;
  static thread_local size_t threadIndex;
  std::atomic<bool> terminate;


  void runThread(unsigned int index) {
    auto config = getConfig();
    randomNumberGenerator.seed( config->seed + index );
    threadIndex = index;
    
    while ( population.size() < config->minPopulationSize ) {
      auto threadConfig = getThreadConfig();
      // spawn individual
      auto [ individual, fitness ] = threadConfig->spawn( this );
      // add individual
      add( individual, fitness );
    }
    
    do {
      auto threadConfig = getThreadConfig();
      if ( randomProbability() < threadConfig->crossoverProbability   ) {
        auto parent1 = threadConfig->parentSelection( this );
        auto parent2 = threadConfig->alternativeParentSelection( this );
        while ( parent1.get() == parent2.get() ) {
          parent2 = threadConfig->alternativeParentSelection( this );
        }
        auto& genome1 = *parent1.get();
        auto& genome2 = *parent2.get();
        auto offspring = threadConfig->incubate( this, threadConfig->crossover( this, genome1, genome2 ) );
        auto fitness = threadConfig->evaluate( this, offspring );
        add( offspring, fitness );
      }
      else {
        auto individual = threadConfig->mutationSelection( this );
        auto& genome = *individual.get();
        auto mutant = threadConfig->incubate( this, threadConfig->mutate( this, genome ) );
        auto fitness = threadConfig->evaluate( this, mutant );
        add( mutant, fitness );
      }
      
      if ( config->termination( this ) ) {
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

