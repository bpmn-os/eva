TEST_CASE("Reconfiguration before run", "[reconfiguration]") {
  EVA::EvolutionaryAlgorithm<Permutation, Values>::ThreadConfig defaultConfig = {
    .spawn = EVA::randomPermutation<Permutation, Values>(5),
    .reproduction = {
      { {EVA::binaryTournamentSelection<Permutation, Values>(), EVA::binaryTournamentSelection<Permutation, Values>()}, EVA::orderedCrossover<Permutation, Values>() },
      { {EVA::randomSelection<Permutation, Values>(), EVA::randomSelection<Permutation, Values>()}, EVA::orderedCrossover<Permutation, Values>() }
    }
  };

  EVA::EvolutionaryAlgorithm<Permutation, Values>::ThreadConfig alternativeConfig = {
    .spawn = EVA::randomPermutation<Permutation, Values>(5),
    .reproduction = {
      { {EVA::randomSelection<Permutation, Values>()}, EVA::randomSwap<Permutation, Values>() }
    }
  };

  EVA::EvolutionaryAlgorithm<Permutation, Values> eva({
    .threads = 2,
    .minPopulationSize = 10,
    .maxPopulationSize = 10,
    .maxSolutionCount = 100,
    .initiationFrequency = 1,
    .threadConfig = defaultConfig,
    .incubate = EVA::constructor<Permutation, Values>(),
    .evaluate = EVA::fitnessFunction<Permutation, Values>()
  });

  REQUIRE_NOTHROW(eva.setThreadConfig(1, alternativeConfig));  // Re-configure thread 1

  // Verify configurations are set correctly
  REQUIRE(eva.getThreadConfig(0)->reproduction.size() == 2); // defaultConfig has 2 reproduction operators
  REQUIRE(eva.getThreadConfig(1)->reproduction.size() == 1); // alternativeConfig has 1 reproduction operator

  REQUIRE_NOTHROW(eva.run());         // Algorithm runs successfully
  REQUIRE(eva.getSolutionCount() >= 10);  // Termination condition respected
}

TEST_CASE("Self-reconfiguration during run", "[reconfiguration]") {
  EVA::EvolutionaryAlgorithm<Permutation, Values>::ThreadConfig defaultConfig = {
    .spawn = EVA::randomPermutation<Permutation, Values>(5),
    .reproduction = {
      { {EVA::binaryTournamentSelection<Permutation, Values>(), EVA::binaryTournamentSelection<Permutation, Values>()}, EVA::orderedCrossover<Permutation, Values>() },
      { {EVA::randomSelection<Permutation, Values>(), EVA::randomSelection<Permutation, Values>()}, EVA::orderedCrossover<Permutation, Values>() }
    }
  };

  EVA::EvolutionaryAlgorithm<Permutation, Values>::ThreadConfig alternativeConfig = {
    .spawn = [&defaultConfig](auto* eva) -> Values {
      eva->setThreadConfig(defaultConfig);  // Reconfigure to defaultConfig
      return EVA::randomPermutation<Permutation, Values>(5)(eva);
    },
    .reproduction = {
      { {EVA::randomSelection<Permutation, Values>()}, EVA::randomSwap<Permutation, Values>() }
    }
  };

  EVA::EvolutionaryAlgorithm<Permutation, Values> eva({
    .threads = 2,
    .minPopulationSize = 10,
    .maxPopulationSize = 10,
    .maxSolutionCount = 100,
    .initiationFrequency = 1,
    .threadConfig = defaultConfig,
    .incubate = EVA::constructor<Permutation, Values>(),
    .evaluate = EVA::fitnessFunction<Permutation, Values>()
  });

  REQUIRE_NOTHROW(eva.setThreadConfig(1, alternativeConfig));  // Re-configure thread 1

  REQUIRE(eva.getThreadConfig(1)->reproduction.size() == 1); // alternativeConfig has 1 reproduction operator
  REQUIRE_NOTHROW(eva.run());
  REQUIRE(eva.getThreadConfig(1)->reproduction.size() == 2); // defaultConfig has 2 reproduction operators
  REQUIRE(eva.getSolutionCount() >= 10);  // Algorithm completed
}

