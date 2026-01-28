# EVA - Evolutionary Algorithm

This C++ header-only library provides an implementation of an evolutionary algorithm for use cases in which genomes only provide a partial representation of individuals in the population and the generation of individuals from a genome is computationally expensive. The implementation uses a multi-threaded queue-based architecture where worker threads generate offspring and a main thread evaluates them and manages the population.

## Architecture

The library uses a **producer-consumer** architecture to minimize lock contention and maximize scalability:

### Worker Threads (Producers)
- **Selection**: Choose parent individuals from population (requires brief lock)
- **Reproduction**: Apply genetic operators (crossover/mutation) to create `Genome`
- **Incubation**: Transform `Genome` into `Individual`
- **Queue**: Add unevaluated `Individual` to thread-safe queue

### Main Thread (Consumer)
- **Evaluation**: Compute fitness for queued individuals
- **Duplicate Detection**: Check if individual already exists in population
- **Population Management**: Insert new individuals or replace worst
- **Termination**: Check stopping conditions and signal workers
- **Monitoring**: Invoke user callbacks with new solutions

### Queue Processing
The main thread processes individuals in batches controlled by `initiationFrequency` (default: 10). This reduces context switching while maintaining responsiveness. Set to 1 for immediate processing (useful for testing with strict termination conditions).

### Benefits
- **Reduced Lock Contention**: Workers only lock population briefly during selection
- **Centralized Evaluation**: All fitness computation happens in one thread
- **Scalability**: Near-linear scaling with thread count
- **Clean Separation**: Genetic operations (workers) vs. population management (main thread)

## Core Concepts - `Individual` and `Genome`

  * **`Genome`**: This type represents the genetic material directly manipulated by genetic operators such as *crossover* and *mutation*.
  * **`Individual`**: This type represents a complete entity within the population. An `Individual` is created by *incubating* a `Genome`. Incubation may comprise the inclusion of additional data not contained in the `Genome` and heuristic variations of the `Genome` , e.g., by education. 

The library requires that `Individual` objects can be implicitly converted to `Genome` objects, as enforced by the `std::is_convertible_v<Individual, Genome>` constraint. This allows genetic operators expecting a `Genome` to directly process `Individual` objects. A custom `incubate` function is needed to transform a `Genome` (the result of genetic operations) into an `Individual` suitable for population inclusion.

## Configuration

The library uses two configuration structures:

### Global Configuration (`Config`)

Controls algorithm-wide settings:

  * `threads`: Number of worker threads
  * `minPopulationSize`: Minimum population size before evolution starts
  * `maxPopulationSize`: Maximum number of individuals to keep
  * `maxSolutionCount`: Stop after generating this many solutions
  * `maxComputationTime`: Maximum runtime in seconds
  * `maxNonImprovingSolutionCount`: Stop after this many solutions without improvement
  * `initiationFrequency`: Process queue when it contains this many pending individuals (default: 10)
  * `threadConfig`: Default configuration for all threads
  * `incubate`: Transforms `Genome` to `Individual`: `Individual(EVA*, const Genome&)`
  * `evaluate`: Assigns fitness to `Individual`: `Fitness(EVA*, const Individual&)`
  * `termination`: Callback to check if algorithm should stop `bool(EVA*)`
  * `monitor`: Callback invoked for each new solution `void(EVA*, const Individual&, const Fitness&)`

> [!IMPORTANT]
> **Threading and Locking:**
> - `termination` is called from the **main thread** after processing each batch of individuals
> - `monitor` is called from the **main thread** while **population is already locked** - safe to call `getPopulation()`, `getBest()`, `getWorst()` without additional locking
> - If callbacks access external variables (captured by reference), protect them with locks or atomics if worker threads might access them

### Thread Configuration (`ThreadConfig`)

Defines genetic operators per thread (each thread can have different strategies):

  * `spawn`: Creates initial genomes `Genome(EVA*)`
  * `adaptationRate`: Rate for adaptive selection of reproduction strategy (currently disabled)
  * `reproduction`: Vector of reproduction strategies, each containing:
    - Selection function: chooses parent(s) `Individual(EVA*)`
    - Parent count: number of parents required
    - Reproduction operator: creates offspring `Genome(EVA*, vector<Individual>&)`
    - Initial weight: starting probability weight

**Adaptive Operator Selection:**
When multiple reproduction strategies are provided, the algorithm uses roulette wheel selection based on operator weights. *(Note: Weight adaptation is currently disabled and under redesign)*
  - The `adaptationRate` parameter is reserved for future use
  - Weights are **thread-local** - each thread can learn independently which operators work best
  - Access current weights via `getWeights()` from within callbacks (only callable from worker threads)


## Workflow

### Initial Population Seeding
1. Worker threads call `spawn()` to generate initial `Genome` objects
2. Workers call `incubate()` to transform genomes into `Individual` objects
3. Workers add unevaluated individuals to queue
4. Main thread evaluates individuals and adds them to population
5. Process continues until `minPopulationSize` is reached

### Evolution Loop
1. **Worker threads** (in parallel):
   - Select parent(s) from population
   - Apply reproduction operator to create offspring `Genome`
   - Incubate `Genome` into `Individual`
   - Add unevaluated individual to queue
   - Check `terminate` flag and exit if set

2. **Main thread**:
   - Wait for queue to reach `initiationFrequency` size
   - Extract batch of individuals from queue
   - For each individual:
     - Evaluate fitness
     - Check for duplicates
     - Insert into population (or replace worst if full)
     - Invoke `monitor` callback
   - Check termination conditions
   - Set `terminate` flag if stopping criteria met
   - Continue until all workers finish and queue is empty

### Termination
- Main thread checks conditions after processing each batch: `maxSolutionCount`, `maxNewSolutionCount`, `maxNonImprovingSolutionCount`, `maxComputationTime`, or custom `termination` callback
- When condition met, main thread sets `terminate = true`
- Workers detect flag and exit their loops
- Main thread processes remaining queued individuals
- Algorithm returns when all workers finish and queue is empty

## Example

An example demonstrating the usage of this library can be found in the `example/` directory.

## Testing

Tests can be run as follows.

**Build and run tests:**
```bash
cd tests
make run
```

## License

MIT licensed

Copyright (C) 2025 Asvin Goel

