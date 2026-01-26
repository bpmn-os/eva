TEST_CASE("Reconfiguration before run", "[reconfiguration]") {
  EVA::EvolutionaryAlgorithm<Permutation, Values>::ThreadConfig defaultConfig = {
    .spawn = EVA::randomPermutation<Permutation, Values>(5),
    .reproduction = {
      { EVA::binaryTournamentSelection<Permutation, Values>(), 2, EVA::orderedCrossover<Permutation, Values>(), 1.0 },
      { EVA::randomSelection<Permutation, Values>(), 2, EVA::orderedCrossover<Permutation, Values>(), 1.0 }
    },
    .incubate = EVA::constructor<Permutation, Values>(),
    .evaluate = EVA::fitnessFunction<Permutation, Values>()
  };

  EVA::EvolutionaryAlgorithm<Permutation, Values>::ThreadConfig alternativeConfig = {
    .spawn = EVA::randomPermutation<Permutation, Values>(5),
    .reproduction = {
      { EVA::randomSelection<Permutation, Values>(), 1, EVA::randomSwap<Permutation, Values>(), 1.0 }  
    },
    .incubate = EVA::constructor<Permutation, Values>(),
    .evaluate = EVA::fitnessFunction<Permutation, Values>()
  };

  EVA::EvolutionaryAlgorithm<Permutation, Values> eva({
    .threads = 2,
    .minPopulationSize = 5,
    .maxPopulationSize = 10,
    .maxSolutionCount = 10,
    .threadConfig = defaultConfig
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
      { EVA::binaryTournamentSelection<Permutation, Values>(), 2, EVA::orderedCrossover<Permutation, Values>(), 1.0 },
      { EVA::randomSelection<Permutation, Values>(), 2, EVA::orderedCrossover<Permutation, Values>(), 1.0 }
    },
    .incubate = EVA::constructor<Permutation, Values>(),
    .evaluate = EVA::fitnessFunction<Permutation, Values>()
  };

  EVA::EvolutionaryAlgorithm<Permutation, Values>::ThreadConfig alternativeConfig = {
    .spawn = [&defaultConfig](auto* eva) -> std::pair<std::shared_ptr<const Permutation>, EVA::Fitness> {
      eva->setThreadConfig(defaultConfig);  // Reconfigure to defaultConfig
      return EVA::randomPermutation<Permutation, Values>(5)(eva);
    },
    .reproduction = {
      { EVA::randomSelection<Permutation, Values>(), 1, EVA::randomSwap<Permutation, Values>(), 1.0 }
    },
    .incubate = EVA::constructor<Permutation, Values>(),
    .evaluate = EVA::fitnessFunction<Permutation, Values>()
  };

  EVA::EvolutionaryAlgorithm<Permutation, Values> eva({
    .threads = 2,
    .minPopulationSize = 10,
    .maxPopulationSize = 10,
    .maxSolutionCount = 10,
    .threadConfig = defaultConfig
  });

  REQUIRE_NOTHROW(eva.setThreadConfig(1, alternativeConfig));  // Re-configure thread 1

  REQUIRE(eva.getThreadConfig(1)->reproduction.size() == 1); // alternativeConfig has 1 reproduction operator
  REQUIRE_NOTHROW(eva.run());
  REQUIRE(eva.getThreadConfig(1)->reproduction.size() == 2); // defaultConfig has 2 reproduction operators
  REQUIRE(eva.getSolutionCount() >= 10);  // Algorithm completed
}

