cmake_minimum_required(VERSION 3.10)
project(compute)

# Engine
set(PSEMEK_EXAMPLES OFF)
add_subdirectory(psemek)

# Subprojects
add_subdirectory(blur)
