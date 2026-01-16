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

/**
 * @brief Multi-threaded evolutionary algorithm with adaptive operator selection
 *
 * A template-based evolutionary algorithm that evolves a population of individuals
 * using configurable genetic operators (selection, crossover, mutation). Supports
 * multi-threading with per-thread configurations and automatically adapts operator
 * selection based on which strategies produce better solutions.
 *
 * @tparam Individual The complete entity in the population (must be convertible to Genome)
 * @tparam Genome The genetic material manipulated by genetic operators
 */
template < typename Individual, typename Genome = Individual >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome>
)
class EvolutionaryAlgorithm {
public:
  /**
   * @brief Configuration for thread-specific evolutionary operators
   *
   * Defines genetic operators (spawn, selection, reproduction) and adaptation settings
   * for a single thread. Threads can have different configurations to explore the
   * search space using different strategies simultaneously.
   *
   * Supports adaptive operator selection: when multiple reproduction strategies are
   * provided, the algorithm learns which work best and uses them more frequently.
   * Controlled by adaptationRate (higher = faster learning).
   */
  struct ThreadConfig {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;

    /// Function to create initial individuals (used during population seeding)
    std::function<std::pair< std::shared_ptr< const Individual >, Fitness >(EVA*)> spawn = nullptr;

    /**
     * @brief Learning rate for adaptive operator selection (0.0 = no learning, 1.0 = instant)
     *
     * When a strategy produces a new best solution, its weight increases by this rate.
     * Default 0.1 provides gradual, stable learning. Higher values (e.g., 0.5) adapt faster
     * but may be less stable.
     */
    double adaptationRate = 0.1;

    /**
     * @brief Reproduction strategies: (selector, num_parents, operator, initial_weight)
     *
     * Multiple strategies can be provided (e.g., crossover, different mutations).
     * The algorithm learns which produce better solutions and uses them more frequently.
     */
    std::vector< std::tuple<
      std::function<std::shared_ptr< const Individual >(EVA*)>, // selection
      size_t, // required individuals
      std::function<Genome(EVA*, const std::vector< std::shared_ptr< const Individual > >&)>, // reproduction
      double // initial weight
    > > reproduction = {};

    /// Function to transform a genome into a complete individual
    std::function<std::shared_ptr< const Individual >(EVA*, const Genome&)> incubate = nullptr;

    /// Function to compute fitness for an individual
    std::function<Fitness(EVA*, const std::shared_ptr< const Individual >&)> evaluate = nullptr;
  };

  struct Config {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;
    unsigned int seed = std::random_device{}(); /// Seed to initialise the random number generators
    unsigned int threads = std::max(1u, std::thread::hardware_concurrency()); /// Number of threads to be used
    size_t minPopulationSize = 10; /// Minimum number of individuals spawned before starting evolutionary process
    size_t maxPopulationSize = 100; /// Maximum number of individuals in the population
    unsigned int maxComputationTime = std::numeric_limits<unsigned int>::max();  /// Time limit in seconds
    unsigned int maxSolutionCount = std::numeric_limits<unsigned int>::max();  /// Maximum number of solutions to be generated before termination 
    unsigned int maxNonImprovingSolutionCount = std::numeric_limits<unsigned int>::max(); /// Maximum number of solutions without improvement to be generated before termination 
    ThreadConfig threadConfig = {}; /// Default configuration for the threads 
    std::function<bool(EVA*)> termination = nullptr; /// Custom termination function
    std::function<void(EVA*, const std::shared_ptr< const Individual >&, const Fitness&)> monitor = nullptr; /// Callback allowing to monitor the progress of the algorithm (note: the population is locked while the callback is executed)
  };

   /// Comparator ordering elements with higher fitness before elements with lower 
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
    if ( config.minPopulationSize < 1 ) {
      throw std::logic_error("EvolutionaryAlgorithm: minimal population size must be at least 1");
    }
    if ( config.maxPopulationSize < config.minPopulationSize ) {
      throw std::logic_error("EvolutionaryAlgorithm: maximal population size must be at least minimal population");
    }

    if ( config.threadConfig.reproduction.empty() ) {
      throw std::logic_error("EvolutionaryAlgorithm: reproduction operator(s) missing");
    }
    if ( !config.threadConfig.incubate ) {
      throw std::logic_error("EvolutionaryAlgorithm: incubator missing");
    }
    if ( !config.threadConfig.evaluate ) {
      throw std::logic_error("EvolutionaryAlgorithm: evaluator missing");
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
    solutionCount++;
    if ( fitness > getBest(true).second ) {
      nonImprovingSolutionCount = 0;      
    }
    else {
      nonImprovingSolutionCount++;
    }
        
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

  [[nodiscard]] std::mt19937& getRandomNumberGenerator() const {
    return randomNumberGenerator;
  }
  
  /// Returns population of individuals with their fitness
  [[nodiscard]] const std::vector< std::pair< std::shared_ptr< const Individual >, Fitness > >& getPopulation() const {  
    return population;
  }
  
  /// Returns a set of population indices, ordered such that individuals with higher fitness appear before those with lower fitness
  [[nodiscard]] const std::set<size_t, Comparator>& getOrderedIndices() const {
    return orderedIndices;
  }

  /**
   * @brief Get current adaptive weights for reproduction strategies
   * @return Normalized probabilities for each strategy (thread-local)
   * @throws std::logic_error if called from main thread (outside worker threads)
   */
  [[nodiscard]] const std::vector<double>& getWeights() const {
    if (getThreadIndex() == 0) {
      throw std::logic_error("getWeights() can only be called from worker threads during run()");
    }
    return weights;
  }

  void run() {
    auto config = getConfig();
    solutionCount = 0;
    nonImprovingSolutionCount = 0;
    if ( config->maxComputationTime == std::numeric_limits<unsigned int>::max() ) {
      terminationTime = std::chrono::time_point<std::chrono::system_clock>::max();
    }
    else {
      terminationTime = std::chrono::system_clock::now()  + std::chrono::seconds(config->maxComputationTime);;
    }
    terminate = false;
    threadsRunning = true;
    std::vector<std::jthread> workers;

    for (unsigned int index = 1; index <= config->threads; ++index) {
      workers.emplace_back(
        [this,index](std::stop_token) {
          runThread(index);
        }
      );
    }
    // jthreads auto-join when destroyed at end of scope
    threadsRunning = false;
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

  /**
   * @brief Get thread configuration
   * @param index Thread index (default: current thread)
   * @return Thread configuration
   */
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

  /**
   * @brief Set configuration for a specific thread (must be called before run())
   * @param index Thread index (0 = global default, 1-N = worker threads)
   * @param config New configuration
   * @throws std::logic_error if called after run() has started
   *
   * For runtime reconfiguration, use setThreadConfig(config) without index.
   */
  void setThreadConfig(size_t index, ThreadConfig config) {
    if (threadsRunning) {
      throw std::logic_error(
        "setThreadConfig(index, config) can only be called before run(). "
        "Use setThreadConfig(config) from within threads to reconfigure at runtime."
      );
    }

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

  /**
   * @brief Reconfigure current thread (can be called during run())
   * @param config New configuration
   *
   * Thread reinitializes its adaptive weights to match the new configuration.
   * Typically used from within callback functions.
   */
  void setThreadConfig(ThreadConfig config, bool reinitializeWeights = false) {
    size_t index = getThreadIndex();
    if (index > 0) {
      std::unique_lock lock(*threadConfigMutex[index-1]);
      threadConfigs[index - 1] = std::make_shared<ThreadConfig>(std::move(config));
      if ( reinitializeWeights ) {
        initializeWeights(threadConfigs[index - 1]);  // Safe - our own weights
      }
    }
    else {
      std::unique_lock lock(globalConfigMutex);
      auto global = std::make_shared<Config>(*globalConfig);  // copy
      global->threadConfig = std::move(config);
      globalConfig = std::move(global);
    }
  }

  static size_t getThreadIndex() { return threadIndex; };
  unsigned int getSolutionCount() const { return solutionCount; };
  unsigned int getNonImprovingSolutionCount() const { return nonImprovingSolutionCount; };
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
  static thread_local bool lockAcquired;
  static thread_local std::vector<double> weights;
  static thread_local double totalWeight;
  std::atomic<unsigned int> solutionCount;
  std::atomic<unsigned int> nonImprovingSolutionCount;
  std::chrono::time_point<std::chrono::system_clock> terminationTime;
  std::atomic<bool> terminate;
  std::atomic<bool> threadsRunning{false};

  void runThread(unsigned int index) {
    auto config = getConfig();
    randomNumberGenerator.seed( config->seed + index );
    threadIndex = index;

    auto threadConfig = getThreadConfig();
    initializeWeights(threadConfig);

    while ( population.size() < config->minPopulationSize ) {
      // update to latest config
      threadConfig = getThreadConfig();
      // spawn individual
      auto [ individual, fitness ] = threadConfig->spawn( this );
      // add individual
      add( individual, fitness );
    }

    do {
      threadConfig = getThreadConfig();
      Fitness fitness;
      auto weightThreshold = randomProbability() * totalWeight;
      double cumulativeWeight = 0.0;
      for ( unsigned int i = 0; i < threadConfig->reproduction.size(); i++ ) {
        auto& [ selector, requiredIndividuals, reproduction, initialWeight ] = threadConfig->reproduction[i];
        // do roulette wheel selection
        cumulativeWeight += weights[i];
        if (cumulativeWeight >= weightThreshold) {
          // create offspring with selected reproduction strategy
          std::vector< std::shared_ptr< const Individual > > individuals;
          individuals.reserve(requiredIndividuals);

          while ( individuals.size() < requiredIndividuals ) {
            auto lock = acquireLock();
            auto individual = selector( this );
            if (
              std::find_if(
                individuals.begin(),
                individuals.end(),
                [&individual](const auto& other) { return other.get() == individual.get(); }
              )
              ==
              individuals.end()
            ) {
              individuals.push_back( individual );
            }
          }

          auto offspring = threadConfig->incubate( this, reproduction( this, individuals ) );
          fitness = threadConfig->evaluate( this, offspring );
          updateWeights( weights[i], threadConfig->adaptationRate, fitness );
          add( offspring, fitness );
          break;
        }
      }
      if (
        solutionCount >= config->maxSolutionCount ||
        nonImprovingSolutionCount >= config->maxNonImprovingSolutionCount ||
        std::chrono::system_clock::now() >= terminationTime ||
        ( config->termination && config->termination( this ) )
      ) {
        terminate = true;
      }
    } while ( !terminate );
  }

  /// Normalize weights to sum to 1.0 (for roulette wheel selection)
  void normalizeWeights() {
    for ( auto& weight : weights ) {
      weight /= totalWeight;
    }
    totalWeight = 1.0;
  }

  /// Initialize thread-local weights from config (called at startup and reconfiguration)
  void initializeWeights(const std::shared_ptr<ThreadConfig>& config) {
    weights.clear();
    totalWeight = 0.0;
    for ( auto& [ selector, quantity, reproduction, weight ] : config->reproduction ) {
      weights.push_back( weight );
      totalWeight += weight;
    }
    normalizeWeights();
  }

  /// Update weights when a strategy produces a new best solution (adaptive learning)
  void updateWeights(double& weight, double adaptationRate, const Fitness& fitness) {
    if ( fitness > getBest().second ) {
      // scale down all weights and increase the successful one
      for ( auto& otherWeight : weights ) {
        otherWeight -= adaptationRate * otherWeight;
      }
      weight += adaptationRate * totalWeight;
    }
// TODO: is it worth penalizing the weight when solution is not improving?
/*
    else {
      // scale down weight and normalize
      totalWeight -= adaptationRate * weight;
      weight *= (1.0 - adaptationRate);
      normalizeWeights();
    }
*/
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

template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome>
)
thread_local bool EvolutionaryAlgorithm<Individual, Genome>::lockAcquired = false;

template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome>
)
thread_local std::vector<double> EvolutionaryAlgorithm<Individual, Genome>::weights = {};

template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome>
)
thread_local double EvolutionaryAlgorithm<Individual, Genome>::totalWeight = 0.0;

} // end namespace EVA

