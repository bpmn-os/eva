# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++23 -pthread -Wall -Wextra -Werror -Wno-error=unused-parameter -Wno-error=unused-but-set-variable -fmax-errors=1

# Source files
SRCS = example.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
TARGET = example

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to clean object files and executable
clean:
	rm -f $(OBJS) $(TARGET)

