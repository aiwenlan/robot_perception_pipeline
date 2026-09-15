cmake_minimum_required(VERSION 3.16)
project(pcl_mini_demo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ubuntu PCL packages pull VTK, which may require MPI::MPI_C even for headless demos.
find_package(MPI COMPONENTS C QUIET)
if(NOT TARGET MPI::MPI_C)
  add_library(MPI::MPI_C INTERFACE IMPORTED)
endif()

find_package(PCL 1.12 REQUIRED COMPONENTS
  common
  io
  filters
  sample_consensus
  segmentation
  search
  kdtree
  registration
)

add_executable(pcl_mini_demo src/main.cpp)
target_include_directories(pcl_mini_demo PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcl_mini_demo PRIVATE ${PCL_LIBRARIES})
if(MPI_C_FOUND)
  target_link_libraries(pcl_mini_demo PRIVATE MPI::MPI_C)
endif()
target_compile_definitions(pcl_mini_demo PRIVATE ${PCL_DEFINITIONS})

message(STATUS "PCL version: ${PCL_VERSION}")
