cmake_minimum_required(VERSION 3.10)
project(compute)

set(CMAKE_CXX_STANDARD 20)

# Engine
set(PSEMEK_EXAMPLES OFF)
add_subdirectory(psemek)

# Subprojects
add_subdirectory(blur)
