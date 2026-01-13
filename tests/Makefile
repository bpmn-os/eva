CXX = g++
CXXFLAGS = -std=c++23 -pthread -Wall -Wextra -Werror -Wno-error=unused-parameter -Wno-error=unused-but-set-variable -I../include
LDFLAGS = -pthread

# Fetch Catch2 v2 single-header (simpler for Makefile projects)
CATCH2_URL = https://raw.githubusercontent.com/catchorg/Catch2/v2.13.10/single_include/catch2/catch.hpp
CATCH2_HEADER = catch.hpp

.PHONY: all run clean

TEST_EXEC = tests

all: $(TEST_EXEC)

$(CATCH2_HEADER):
	@echo "Downloading Catch2..."
	@wget -q $(CATCH2_URL) -O $(CATCH2_HEADER) || curl -s $(CATCH2_URL) -o $(CATCH2_HEADER)

$(TEST_EXEC): main.cpp sorting.cpp reconfiguration.cpp adaptiveness.cpp $(CATCH2_HEADER)
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

run: $(TEST_EXEC)
	./$(TEST_EXEC)

clean:
	rm -f $(TEST_EXEC) $(CATCH2_HEADER)
