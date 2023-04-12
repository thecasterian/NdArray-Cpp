INCLUDE_DIR = include
INCLUDES = $(wildcard $(INCLUDE_DIR)/*)

.PHONY: all clean

all: main

main: main.cpp $(INCLUDES)
	g++ -std=gnu++23 -Wall -Wextra -Wpedantic -Werror=comma-subscript -Iinclude -o main main.cpp

clean:
	rm -f main
