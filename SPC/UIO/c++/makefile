# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -g

# Sources and objects
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

# Output executable
TARGET = program.exe

# Default rule
all: $(TARGET)

# Rule to link objects and create the executable
$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $^

# Rule to compile source files into objects
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean the build
clean:
	del /Q *.o $(TARGET)

# Phony targets
.PHONY: all clean
