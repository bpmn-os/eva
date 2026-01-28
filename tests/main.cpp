#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// EVA library
#include <eva.h>
#include <spawn/randomPermutation.h>
#include <selection/randomSelection.h>
#include <selection/binaryTournamentSelection.h>
#include <crossover/orderedCrossover.h>
#include <mutator/randomSwap.h>
#include <incubator/constructor.h>
#include <evaluator/fitnessFunction.h>
#include <calibration/weightUpdate.h>

// Test problem: sort a permutation of integers
// Genome type represents the genetic material (vector of integers)
using Values = std::vector<unsigned int>;

// Individual type that can be evaluated and converted to Genome
struct Permutation {
  Values values;

  // Fitness: ratio of adjacent pairs in ascending order (1.0 = sorted)
  [[nodiscard]] EVA::Fitness getFitness() const {
    if (values.empty()) return {0.0};
    int count = 0;
    for (size_t i = 1; i < values.size(); ++i) {
      if (values[i - 1] <= values[i]) ++count;
    }
    return {static_cast<double>(count) / (values.size() - 1)};
  }

  // Constructor from Genome (required by incubator)
  Permutation(Values v) : values(std::move(v)) {}
  // Conversion to Genome (required by genetic operators)
  operator const Values&() const { return values; }

  // Equality comparison
  bool operator==(const Permutation& other) const {
    return values == other.values;
  }
};

// Include all test files
#include "sorting.cpp"
#include "reconfiguration.cpp"
#include "calibration.cpp"
