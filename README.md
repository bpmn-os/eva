# EVA - Evolutionary Algorithm

This C++ header-only library provides an implementation of an evolutionary algorithm for use cases in which genomes only provide a partial representation of individuals in the population and the generation of individuals from a genome is computationally expensive. The implementation is multi-threaded and each new individual created is immediately added to the population, directly replacing the worst individual.

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
  * `threadConfig`: Default configuration for all threads
  * `termination`: Callback to check if algorithm should stop `bool(EVA*)`
  * `monitor`: Callback invoked for each new solution `void(EVA*, const Individual&, const Fitness&)`

> [!IMPORTANT]
> **Threading and Locking:**
> - Both `termination` and `monitor` are **called from worker threads**
> - If you access your own variables (captured by reference), protect them with locks or atomics
> - `monitor` is called while **population is already locked** - safe to call `getPopulation()`, `getBest()`, `getWorst()` without additional locking

### Thread Configuration (`ThreadConfig`)

Defines genetic operators per thread (each thread can have different strategies):

  * `spawn`: Creates initial individuals `pair<Individual, Fitness>(EVA*)`
  * `adaptationRate`: Rate for adaptive selection of reproduction strategy
  * `reproduction`: Vector of reproduction strategies, each containing:
    - Selection function: chooses parent(s) `Individual(EVA*)`
    - Parent count: number of parents required
    - Reproduction operator: creates offspring `Genome(EVA*, vector<Individual>&)`
    - Initial weight: starting probability weight
  * `incubate`: Transforms `Genome` to `Individual`: `Individual(EVA*, const Genome&)`
  * `evaluate`: Assigns fitness to `Individual`: `Fitness(EVA*, const Individual&)`

**Adaptive Operator Selection:**
When multiple reproduction strategies are provided, the algorithm uses roulette wheel selection based on operator weights. The weights are updated after each reproduction based on the offspring's fitness:
  - Successful operators (producing fit offspring) get their weights increased
  - The `adaptationRate` controls how quickly weights adapt (0.0 = no adaptation)
  - Weights are **thread-local** - each thread learns independently which operators work best
  - Access current weights via `getWeights()` from within callbacks (only callable from worker threads)


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

