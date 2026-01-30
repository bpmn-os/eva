#pragma once

#include <vector>
#include <set>
#include <deque>
#include <utility>
#include <concepts>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
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
  std::is_convertible_v<Individual,Genome> &&                                 
  std::equality_comparable<Individual>     
)
class EvolutionaryAlgorithm {
public:
  /**
   * @brief Configuration for thread-specific evolutionary operators
   *
   * Defines genetic operators (spawn, selection, reproduction) and calibration settings
   * for a single thread. Threads can have different configurations to explore the
   * search space using different strategies simultaneously.
   *
   * Supports adaptive operator selection: when multiple reproduction strategies are
   * provided, the algorithm learns which work best and uses them more frequently.
   */
  struct ThreadConfig {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;

    /// Function to create initial genomes (used during population seeding)
    std::function<Genome(EVA*)> spawn = nullptr;

    /**
     * @brief Reproduction operators: (selectors, operator, initial_weight)
     *
     * Multiple operators can be provided (e.g., crossover, different mutations).
     * The algorithm learns which produce better solutions and uses them more frequently.
     */
    std::vector< std::tuple<
      std::vector<std::function<std::shared_ptr< const Individual >(EVA*)>>, // selectors (one per parent)
      std::function<Genome(EVA*, const std::vector< std::shared_ptr< const Individual > >&)> // reproduction
    > > reproduction = {};

    /// Adaptive learning callback: updates weights based on offspring feedback
    std::function<void(EVA*, const std::shared_ptr<const Individual>&, size_t reproducer, const Fitness&, bool isUnfit, bool isDuplicate, bool isFittest)> calibration = nullptr;
  };

  struct Config {
    using EVA = EvolutionaryAlgorithm<Individual, Genome>;
    unsigned int seed = std::random_device{}(); /// Seed to initialise the random number generators
    unsigned int threads = std::max(1u, std::thread::hardware_concurrency()); /// Number of threads to be used
    size_t minPopulationSize = 10; /// Minimum number of individuals spawned before starting evolutionary process
    size_t maxPopulationSize = 100; /// Maximum number of individuals in the population
    unsigned int maxComputationTime = std::numeric_limits<unsigned int>::max();  /// Time limit in seconds
    unsigned int maxSolutionCount = std::numeric_limits<unsigned int>::max();  /// Maximum number of solutions to be generated before termination 
    unsigned int maxNewSolutionCount = std::numeric_limits<unsigned int>::max();  /// Maximum number of non-duplicate solutions to be generated before termination 
    unsigned int maxNonImprovingSolutionCount = std::numeric_limits<unsigned int>::max(); /// Maximum number of solutions without improvement to be generated before termination
    size_t initiationFrequency = 1; /// Process queue when it contains this many pending individuals
    ThreadConfig threadConfig = {}; /// Default configuration for the threads
    /// Function to transform a genome into a complete individual
    std::function<std::shared_ptr< const Individual >(EVA*, const Genome&)> incubate = nullptr;
    /// Function to compute fitness for an individual
    std::function<Fitness(EVA*, const std::shared_ptr< const Individual >&)> evaluate = nullptr;
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
    if ( 
      auto& [selectors, reproduction] = config.threadConfig.reproduction.back(); 
      selectors.size() > config.minPopulationSize 
    ) {
      throw std::logic_error("EvolutionaryAlgorithm: last reproduction operator must not have more selectors then minimal population size");
    }       
    
    if ( !config.incubate ) {
      throw std::logic_error("EvolutionaryAlgorithm: incubator missing");
    }
    if ( !config.evaluate ) {
      throw std::logic_error("EvolutionaryAlgorithm: evaluator missing");
    }
    
    threadConfigMutex.resize(config.threads);
    threadConfigs.resize(config.threads);
    initiatedOffspring.resize(config.threads);
    initiatedOffspringMutexes.resize(config.threads);
    for (size_t i = 0; i < config.threads; ++i) {
      threadConfigMutex[i] = std::make_unique< std::shared_mutex >();
      threadConfigs[i] = std::make_shared<ThreadConfig>(config.threadConfig);
      initiatedOffspringMutexes[i] = std::make_unique< std::mutex >();
    }

    globalConfig = std::make_shared<Config>(std::move(config));
    
  };

  /// Adds an unevaluated individual to the queue for population management
  void add( std::shared_ptr< const Individual > individual, size_t threadIndex ) {
    {
      std::lock_guard lock(queueMutex);
      pendingOffspring.emplace_back(std::move(individual), threadIndex);
    }
    // Always notify main thread (let it decide whether to process)
    pendingWork.notify_one();
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

  // Thread-local weights
  static thread_local std::vector<double> weights;
  static thread_local double totalWeight;

  // Thread-local stats
  static thread_local std::vector<std::tuple<unsigned int,unsigned int,unsigned int>> stats; // statistics for each reproducer (count,insertions,improvements)

  /// Normalize weights to sum to 1.0 (for roulette wheel selection)
  void normalizeWeights() {
    for ( auto& weight : weights ) {
      weight /= totalWeight;
    }
    totalWeight = 1.0;
  }

  /// Initialize thread-local weights from config
  void initializeWeights(const std::shared_ptr<ThreadConfig>& config) {
    weights.clear();
    for ( unsigned int i = 0; i < config->reproduction.size(); i++ ) {
      weights.push_back( 1.0 / config->reproduction.size() );
    }
    totalWeight = 1.0;
  }

  /// Initialize thread-local stats
  void initializeStats(const std::shared_ptr<ThreadConfig>& config) {
    stats.clear();
    for ( unsigned int i = 0; i < config->reproduction.size(); i++ ) {
      stats.push_back( {0, 0, 0} );
    }
  }

  void run() {
    auto config = getConfig();
    solutionCount = 0;
    newSolutionCount = 0;
    nonImprovingSolutionCount = 0;
    if ( config->maxComputationTime == std::numeric_limits<unsigned int>::max() ) {
      terminationTime = std::chrono::time_point<std::chrono::system_clock>::max();
    }
    else {
      terminationTime = std::chrono::system_clock::now()  + std::chrono::seconds(config->maxComputationTime);;
    }
    terminate = false;
    activeWorkers = config->threads;
    std::vector<std::jthread> workers;

    for (unsigned int index = 1; index <= config->threads; ++index) {
      workers.emplace_back(
        [this,index](std::stop_token) {
          runThread(index);
          // Worker finished - decrement counter and notify
          activeWorkers--;
          pendingWork.notify_one();
        }
      );
    }

    // Main thread becomes queue manager
    while (activeWorkers > 0 || !pendingOffspring.empty()) {
      std::unique_lock lock(queueMutex);

      // Wait until queue reaches batch size or all workers finish
      pendingWork.wait(lock, [this, &config]() {
        return pendingOffspring.size() >= config->initiationFrequency || activeWorkers == 0;
      });

      // Extract all queued entries (already holding lock)
      std::deque<std::pair<std::shared_ptr<const Individual>, size_t>> novices;
      novices.swap(pendingOffspring);  // O(1) swap, pendingOffspring becomes empty
      lock.unlock();

      // Process novices one-by-one
      while (!novices.empty()) {
        auto [individual, threadIdx] = std::move(novices.front());
        novices.pop_front();
        initiate(individual, threadIdx);
      }

      // Check termination conditions (main thread has accurate counters)
      if (
        solutionCount >= config->maxSolutionCount ||
        newSolutionCount >= config->maxNewSolutionCount ||
        nonImprovingSolutionCount >= config->maxNonImprovingSolutionCount ||
        std::chrono::system_clock::now() >= terminationTime ||
        (config->termination && config->termination(this))
      ) {
        terminate = true;
      }
    }

    // Workers auto-join when jthreads are destroyed
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
    if (activeWorkers > 0) {
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
  void setThreadConfig(ThreadConfig config) {
    size_t index = getThreadIndex();
    if (index > 0) {
      std::unique_lock lock(*threadConfigMutex[index-1]);
      threadConfigs[index - 1] = std::make_shared<ThreadConfig>(std::move(config));
      initializeWeights(threadConfigs[index - 1]);  // Always reinitialize weights
      initializeStats(threadConfigs[index - 1]);  // Always reinitialize stats
      createdOffspring.clear();
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
  unsigned int getNewSolutionCount() const { return newSolutionCount; };
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
  static thread_local std::deque<std::pair<std::shared_ptr<const Individual>, size_t>> createdOffspring;
  std::atomic<unsigned int> solutionCount;
  std::atomic<unsigned int> newSolutionCount;
  std::atomic<unsigned int> nonImprovingSolutionCount;
  std::chrono::time_point<std::chrono::system_clock> terminationTime;
  std::atomic<bool> terminate;
  // Queue for pending unevaluated individuals with their source thread
  std::deque<std::pair<std::shared_ptr<const Individual>, size_t>> pendingOffspring;
  mutable std::mutex queueMutex;
  std::condition_variable pendingWork;
  std::atomic<size_t> activeWorkers{0};
  // Per-thread queues of initiated offspring: (individual, fitness, isUnfit, isDuplicate, isFittest)
  std::vector<std::deque<std::tuple<std::shared_ptr<const Individual>, Fitness, bool, bool, bool>>> initiatedOffspring;
  std::vector<std::unique_ptr<std::mutex>> initiatedOffspringMutexes;

  void initiate(const std::shared_ptr<const Individual>& individual, size_t threadIdx) {
    auto config = getConfig();

    // Evaluate individual (main thread responsibility)
    Fitness fitness = config->evaluate(this, individual);

    auto lock = acquireLock();  // Lock population

    solutionCount++;

    // Duplicate checking
    bool isDuplicate = false;
    if (fitness >= getWorst(true).second && fitness <= getBest(true).second) {
      for (auto& [other_individual, other_fitness] : population) {
        if (other_fitness == fitness && *other_individual == *individual) {
          isDuplicate = true;
          break;
        }
      }
    }

    if (!isDuplicate) {
      newSolutionCount++;
    }
    
    bool isUnfit = (fitness < getWorst(true).second && population.size() >= config->maxPopulationSize);
    bool isFittest = (fitness > getBest(true).second);
    
    if (isFittest) {
      nonImprovingSolutionCount = 0;
    }
    else {
      nonImprovingSolutionCount++;
    }

    // Monitor callback
    if (config->monitor) {
      config->monitor(this, individual, fitness);
    }

    // Insert into population
    if (!isUnfit && !isDuplicate) {
      if (population.size() < config->maxPopulationSize) {
        size_t index = population.size();
        population.emplace_back(individual, fitness);
        orderedIndices.insert(index);
      }
      else {
        size_t index = *orderedIndices.rbegin(); // take index of worst individual
        population[index] = std::make_pair(individual, fitness);
        orderedIndices.erase(std::prev(orderedIndices.end()));
        orderedIndices.insert(index);
      }
    }

    // Publish feedback to worker thread
    if (threadIdx > 0) {
      std::lock_guard lock(*initiatedOffspringMutexes[threadIdx - 1]);
      initiatedOffspring[threadIdx - 1].emplace_back(individual, fitness, isUnfit, isDuplicate, isFittest);
    }
  }

  void processFeedback() {
    if (threadIndex == 0) {
      throw std::logic_error("processFeedback() must not be called from main thread.");    
    }

    auto threadConfig = getThreadConfig();
    std::lock_guard lock(*initiatedOffspringMutexes[threadIndex - 1]);
    auto& feedback = initiatedOffspring[threadIndex - 1];

    while (!feedback.empty()) {
      if ( createdOffspring.empty() ) {
        // No offspring waiting for feedback
        feedback.pop_front();
        continue;
      }

      auto& [offspring, reproducer] = createdOffspring.front();
      auto& [novice, fitness, isUnfit, isDuplicate, isFittest] = feedback.front();

      if (offspring.get() != novice.get()) {
        // First offspring waiting for feedback doesn't match novice for which feedback is given - skip this feedback
        feedback.pop_front();
        continue;
      }

      // Update stats
      auto& [count,insertions,improvements] = stats[reproducer];
      count++;
      if ( !isUnfit && !isDuplicate ) insertions++;
      if ( isFittest ) improvements++;      

      // Call calibration callback with feedback info
      if (threadConfig->calibration) {
        threadConfig->calibration(this, offspring, reproducer, fitness, isUnfit, isDuplicate, isFittest);
      }

      createdOffspring.pop_front();
      feedback.pop_front();
    }
  }

  void runThread(unsigned int index) {
    auto config = getConfig();
    randomNumberGenerator.seed( config->seed + index );
    threadIndex = index;

    auto threadConfig = getThreadConfig();
    initializeWeights(threadConfig);
    initializeStats(threadConfig);

    while ( population.size() < config->minPopulationSize && !terminate ) {
      // update to latest config
      threadConfig = getThreadConfig();
      // spawn genome
      auto genome = threadConfig->spawn( this );
      // incubate into individual
      auto individual = config->incubate( this, genome );
      // add individual (main thread will evaluate, no tracking - nothing to learn from spawn)
      add( individual, threadIndex );
    }

    do {
      threadConfig = getThreadConfig();
      auto weightThreshold = randomProbability() * totalWeight;
      double cumulativeWeight = 0.0;
      for ( unsigned int i = 0; i < threadConfig->reproduction.size(); i++ ) {
        auto& [ selectors, reproduction ] = threadConfig->reproduction[i];
        // do roulette wheel selection
        cumulativeWeight += weights[i];
        if (cumulativeWeight >= weightThreshold) {
          if ( selectors.size() > population.size() ) {
            // Population size is too small to find enough individuals
            continue;
          }
        
        
          // create offspring with selected reproduction operator
          std::vector< std::shared_ptr< const Individual > > individuals;
          individuals.reserve(selectors.size());

          for ( unsigned int j = 0; j < selectors.size(); j++ ) {
            auto lock = acquireLock();
            auto individual = selectors[j]( this );
            if (
              std::find_if(
                individuals.begin(),
                individuals.end(),
                [&individual](const auto& other) { return other.get() == individual.get(); }
              )
              !=
              individuals.end()
            ) {
              break; // same individual is selected twice
            }
            individuals.push_back( individual );
          }

          if (individuals.size() == selectors.size()) {
            auto offspring = config->incubate( this, reproduction( this, individuals ) );
            // Track created offspring with reproducer index
            createdOffspring.emplace_back(offspring, i);
            // Add to main queue (main thread will evaluate)
            add( offspring, threadIndex );
            // Process feedback opportunistically
            processFeedback();
            break;
          }
        }
      }
      // Termination checking removed - main thread handles this
    } while ( !terminate );
  }
};


template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome> &&                                 
  std::equality_comparable<Individual>
)
thread_local std::mt19937 EvolutionaryAlgorithm<Individual, Genome>::randomNumberGenerator;

template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome> &&                                 
  std::equality_comparable<Individual>     
)
thread_local size_t EvolutionaryAlgorithm<Individual, Genome>::threadIndex = 0;

template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome> &&                                 
  std::equality_comparable<Individual>     
)
thread_local bool EvolutionaryAlgorithm<Individual, Genome>::lockAcquired = false;

template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome> &&                                 
  std::equality_comparable<Individual>     
)
thread_local std::vector<double> EvolutionaryAlgorithm<Individual, Genome>::weights = {};

template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome> &&                                 
  std::equality_comparable<Individual>     
)
thread_local std::vector<std::tuple<unsigned int,unsigned int,unsigned int>> EvolutionaryAlgorithm<Individual, Genome>::stats = {};

template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome> &&
  std::equality_comparable<Individual>
)
thread_local double EvolutionaryAlgorithm<Individual, Genome>::totalWeight = 0.0;

template < typename Individual, typename Genome >
requires (
  std::movable<Individual> &&
  std::movable<Genome> &&
  std::is_convertible_v<Individual,Genome> &&
  std::equality_comparable<Individual>
)
thread_local std::deque<std::pair<std::shared_ptr<const Individual>, size_t>> EvolutionaryAlgorithm<Individual, Genome>::createdOffspring = {};

} // end namespace EVA

