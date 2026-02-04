cmake_minimum_required(VERSION 3.15)
project(mimosa_core LANGUAGES CXX)

# Find Python components first
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Define the extension module
# Note: Paths are relative to the project root where CMakeLists.txt resides
nanobind_add_module(_core
    src/mimosa/_core/bindings.cpp
    src/mimosa/_core/mco_prc.cpp
)

# Include directories
target_include_directories(_core PRIVATE src/mimosa/_core)

# Set C++ standard
target_compile_features(_core PUBLIC cxx_std_17)

# Enable docstrings in nanobind
target_compile_definitions(_core PRIVATE NB_DOC_STRINGS)

# Install the extension into the package directory
install(TARGETS _core LIBRARY DESTINATION mimosa/_core)