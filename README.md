# EVA - Evolutionary Algorithm

This C++ header-only library provides an implementation of an evolutionary algorithm for use cases in which genomes only provide a partial representation of individuals in the population and the generation of individuals from a genome is computationally expensive. The implementation is multi-threaded and each new individual created is immediately added to the population, directly replacing the worst individual.

## Core Concepts - `Individual` and `Genome`

  * **`Genome`**: This type represents the genetic material directly manipulated by genetic operators such as crossover and mutation.
  * **`Individual`**: This type represents a complete entity within the population. An `Individual` is created by incubating a `Genome`. Incubation may comprise the inclusion of additional data not contained in the `Genome` and heuristic variations of the `Genome` , e.g., by education. 

The library requires that `Individual` objects can be implicitly converted to `Genome` objects, as enforced by the `std::is_convertible_v<Individual, Genome>` constraint. This allows genetic operators expecting a `Genome` to directly process `Individual` objects. An custom `incubate` function is needed to transform a `Genome` (the result of genetic operations) into an `Individual` suitable for population inclusion.

## Features

  * **Generic Design:** The library is templated on `Individual` and `Genome` types, supporting custom data structures for these representations.
  * **Configurable Operators:** Genetic operators like selection, crossover, mutation, as well as other functions for incubation, evaluation, termination, and monitoring are managed via customizable `std::function` objects.

### Configurable Functions

The `EvolutionaryAlgorithm::Config` struct defines `std::function` members for specifying custom functions:

  * `parentSelection`, `alternativeParentSelection`: Functions for selecting parent individuals for reproduction.
  * `crossover`: A function that combines two `Genome` objects to produce a new `Genome`.
  * `mutationSelection`: A function that selects an individual to be mutated.
  * `mutate`: A function that modifies a `Genome` to produce a new `Genome`.
  * `incubate`: A function that transforms a `Genome` into an `Individual`.
  * `evaluate`: A function responsible for assigning a `Fitness` value (represented as `std::vector<double>`) to an `Individual`.
  * `termination`: A function that defines the stopping condition for the evolutionary process.
  * `monitor`: An optional callback function for observing the population or individuals during execution.

-----

## Example

A detailed example demonstrating the usage of this library can be found in the `example/` directory.

-----

## License

MIT licensed

Copyright (C) 2025 Asvin Goel

