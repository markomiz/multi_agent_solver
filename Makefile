# Build directories
BUILD_DIR = build
DEBUG_DIR = $(BUILD_DIR)/debug
RELEASE_DIR = $(BUILD_DIR)/release
PERF_DIR = $(BUILD_DIR)/perf
SRC_DIR = code
EXE_NAME = dynamic_solver

# Default target (release build)
all: release

# Debug build using CMake
debug:
	@mkdir -p $(DEBUG_DIR)
	@cd $(DEBUG_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug ../../$(SRC_DIR)
	cmake --build $(DEBUG_DIR)

# Release build using CMake
release:
	@mkdir -p $(RELEASE_DIR)
	@cd $(RELEASE_DIR) && cmake -DCMAKE_BUILD_TYPE=Release ../../$(SRC_DIR)
	cmake --build $(RELEASE_DIR)

# Release build with debug symbols (for perf analysis)
release_debug:
	@mkdir -p $(PERF_DIR)
	@cd $(PERF_DIR) && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../../$(SRC_DIR)
	cmake --build $(PERF_DIR)

# Run the release version
run: release
	@echo "🚀 Running $(EXE_NAME) in release mode..."
	$(RELEASE_DIR)/$(EXE_NAME)

# Run the debug version with GDB
test: debug
	@echo "🐞 Running $(EXE_NAME) in debug mode with GDB..."
	gdb --args $(DEBUG_DIR)/$(EXE_NAME)

# Run performance profiling using perf
performance: release_debug
	@echo "📊 Running performance analysis using perf..."
	@perf record -g -- $(PERF_DIR)/$(EXE_NAME)
	@perf report

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all debug release release_debug run test performance clean
