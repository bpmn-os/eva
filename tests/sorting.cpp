TEST_CASE("Sorting", "[sorting]") {
  // Configure evolutionary algorithm
  EVA::EvolutionaryAlgorithm<Permutation, Values> eva({
    .threads = 1,                   // Single-threaded for deterministic testing
    .minPopulationSize = 5,         // Start evolution after 5 individuals
    .maxPopulationSize = 10,        // Keep best 10 individuals
    .maxSolutionCount = 50,         // Stop after 50 solutions generated
    .initiationFrequency = 1,       // Process each individual immediately for accurate termination
    .threadConfig = {
      .spawn = EVA::randomPermutation<Permutation, Values>(5),  // Create random 5-element permutations
      .reproduction = {
        {
          {EVA::binaryTournamentSelection<Permutation, Values>(), EVA::binaryTournamentSelection<Permutation, Values>()},  // 2 parents
          EVA::orderedCrossover<Permutation, Values>()          // Crossover operator
        },
      },
    },
    .incubate = EVA::constructor<Permutation, Values>(),       // Genome → Individual
    .evaluate = EVA::fitnessFunction<Permutation, Values>()    // Use Individual::getFitness()
  });

  // Run algorithm
  REQUIRE_NOTHROW(eva.run());

  // Verify results
  auto [best, fitness] = eva.getBest();
  REQUIRE(best != nullptr);              // Found a solution
  REQUIRE(fitness.size() == 1);          // Single objective
  REQUIRE(fitness[0] >= 0.0);            // Valid fitness range
  REQUIRE(fitness[0] <= 1.0);
  auto solutionCount = eva.getSolutionCount();
  REQUIRE(solutionCount >= 50); // At least 50 solutions must be generated
}
