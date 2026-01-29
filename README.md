# EVA - Evolutionary Algorithm

This C++ header-only library provides an implementation of an evolutionary algorithm for use cases in which genomes only provide a partial representation of individuals in the population and the generation of individuals from a genome is computationally expensive. The implementation uses a multi-threaded queue-based architecture where worker threads generate offspring and a main thread evaluates them and manages the population.

## Core Concepts - `Individual` and `Genome`

  * **`Genome`**: This type represents the genetic material directly manipulated by genetic operators such as *crossover* and *mutation*.
  * **`Individual`**: This type represents a complete entity within the population. An `Individual` is created by *incubating* a `Genome`. Incubation may comprise the inclusion of additional data not contained in the `Genome` and heuristic variations of the `Genome`, e.g., by education. 

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
  * `initiationFrequency`: Process queue when it contains this many pending individuals
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

Defines genetic operators per thread (each thread can have different operators):

  * `spawn`: Creates initial genomes `Genome(EVA*)`
  * `reproduction`: Vector of reproduction operators, each containing:
    - Selectors: vector of selection functions, one per parent `vector<function<Individual(EVA*)>>`
    - Reproduction operator: creates offspring `Genome(EVA*, vector<Individual>&)`
  * `calibration`: Callback invoked to update weights for roulette wheel selection  of reproduction operators. 

## Functionality

The library uses a producer-consumer architecture where worker threads generate offspring and a main thread manages the population. During initialization, worker threads call `spawn()` to generate initial genomes, incubate them into individuals, and add them to a thread-safe queue. The main thread evaluates these individuals and inserts them into the population until `minPopulationSize` is reached.

During evolution, each worker thread selects parent individuals from the population, applies a reproduction operator to create an offspring genome, incubates the genome into an individual, tracks the created offspring with its reproducer index, and adds the unevaluated individual to the queue. 

The main thread waits for the queue to reach `initiationFrequency` individuals, then extracts the new individual(s). For each individual, the main thread evaluates fitness and checks whether a duplicate already exists in the population and invokes the `monitor` callback if provided. If the individual is not a duplicate, it is inserted into the population and replaces the worst if `maxPopulationSize` would otherwise be exceeded. Thereafter feedback is provided to the worker thread that created the individual, allowing the worker thread to update its own parameters by invoking its `calibration` callback to adjusts operator weights.

After processing new individuals, the main thread checks termination conditions including `maxSolutionCount`, `maxNewSolutionCount`, `maxNonImprovingSolutionCount`, `maxComputationTime`, and the `termination` callback. When a termination condition is met, the main thread sets a terminate flag. Workers detect this flag and exit their loops. The main thread continues processing remaining queued individuals until all workers finish and the queue is empty.

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

