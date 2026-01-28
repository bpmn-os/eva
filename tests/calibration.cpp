TEST_CASE("Adaptive weights favor successful operators", "[calibration]") {
  auto weights = std::vector<double>{1.0, 1.0};

  // Custom spawn: creates reverse-ordered permutations (worst fitness)
  auto spawnReverse = []([[maybe_unused]] auto* eva) -> Values {
    Values values(5);
    std::iota(values.rbegin(), values.rend(), 0);  // [4,3,2,1,0]
    return values;
  };

  auto goodOperator = []([[maybe_unused]] auto* eva, const std::vector<std::shared_ptr<const Permutation>>& parents) -> Values {
    Values values = parents[0]->values;
    // Swap first two if out of order - makes it better
    for (size_t i = 0; i < values.size()-1; i++) {
      if (values.size() >= 2 && values[i] > values[i+1]) {
        std::swap(values[i], values[i+1]);
      }
    }
    return values;
  };

  auto badOperator = []([[maybe_unused]] auto* eva, const std::vector<std::shared_ptr<const Permutation>>& parents) -> Values {
    Values values = parents[0]->values;
    // Swap first two if in order - makes it worse
    for (size_t i = 0; i < values.size()-1; i++) {
      if (values.size() >= 2 && values[i] < values[i+1]) {
        std::swap(values[i], values[i+1]);
      }
    }
    return values;
  };

  EVA::EvolutionaryAlgorithm<Permutation, Values> eva({
    .threads = 1,
    .minPopulationSize = 1,
    .maxPopulationSize = 10,
    .maxNonImprovingSolutionCount = 100,
    .initiationFrequency = 1,
    .threadConfig = {
      .spawn = spawnReverse,
      .reproduction = {
        { EVA::randomSelection<Permutation, Values>(), 1, goodOperator, 1.0 },  // Good
        { EVA::randomSelection<Permutation, Values>(), 1, badOperator, 1.0 }    // Bad
      },
      .calibration = [&weights](auto* eva, const auto& offspring, size_t reproducer, const auto& fitness, bool isDuplicate, bool isFittest) {
        // Apply default weight update with learning rate 0.1
        EVA::weightUpdate<Permutation, Values>(0.1)(eva, offspring, reproducer, fitness, isDuplicate, isFittest);
        // Capture weights for test verification
        weights = EVA::EvolutionaryAlgorithm<Permutation, Values>::weights;
      }
    },
    .incubate = EVA::constructor<Permutation, Values>(),
    .evaluate = EVA::fitnessFunction<Permutation, Values>()
  });

  REQUIRE_NOTHROW(eva.run());

  // Good operator should have higher weight than bad operator
  REQUIRE(weights[0] > weights[1]);
}

TEST_CASE("Zero adaptation rate keeps weights constant", "[calibration]") {
  auto weights = std::vector<double>{1.0, 1.0};

  // Custom spawn: creates reverse-ordered permutations (worst fitness)
  auto spawnReverse = []([[maybe_unused]] auto* eva) -> Values {
    Values values(5);
    std::iota(values.rbegin(), values.rend(), 0);  // [4,3,2,1,0]
    return values;
  };

  auto goodOperator = []([[maybe_unused]] auto* eva, const std::vector<std::shared_ptr<const Permutation>>& parents) -> Values {
    Values values = parents[0]->values;
    // Swap first two if out of order - makes it better
    for (size_t i = 0; i < values.size()-1; i++) {
      if (values.size() >= 2 && values[i] > values[i+1]) {
        std::swap(values[i], values[i+1]);
      }
    }
    return values;
  };

  auto badOperator = []([[maybe_unused]] auto* eva, const std::vector<std::shared_ptr<const Permutation>>& parents) -> Values {
    Values values = parents[0]->values;
    // Swap first two if in order - makes it worse
    for (size_t i = 0; i < values.size()-1; i++) {
      if (values.size() >= 2 && values[i] < values[i+1]) {
        std::swap(values[i], values[i+1]);
      }
    }
    return values;
  };

  EVA::EvolutionaryAlgorithm<Permutation, Values> eva({
    .threads = 1,
    .minPopulationSize = 1,
    .maxPopulationSize = 10,
    .maxNonImprovingSolutionCount = 100,
    .initiationFrequency = 1,
    .threadConfig = {
      .spawn = spawnReverse,
      .reproduction = {
        { EVA::randomSelection<Permutation, Values>(), 1, goodOperator, 1.0 },  // Good
        { EVA::randomSelection<Permutation, Values>(), 1, badOperator, 1.0 }    // Bad
      },
      .calibration = [&weights](auto* eva, const auto& offspring, size_t reproducer, const auto& fitness, bool isDuplicate, bool isFittest) {
        // Apply default weight update with zero learning rate (should not change weights)
        EVA::weightUpdate<Permutation, Values>(0.0)(eva, offspring, reproducer, fitness, isDuplicate, isFittest);
        // Capture weights for test verification
        weights = EVA::EvolutionaryAlgorithm<Permutation, Values>::weights;
      }
    },
    .incubate = EVA::constructor<Permutation, Values>(),
    .evaluate = EVA::fitnessFunction<Permutation, Values>()
  });

  REQUIRE_NOTHROW(eva.run());

  // Normalised weights should not change with zero adaptation rate
  REQUIRE(weights[0] == 0.5 );
  REQUIRE(weights[1] == 0.5 );
}
