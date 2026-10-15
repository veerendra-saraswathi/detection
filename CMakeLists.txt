cmake_minimum_required(VERSION 3.10)

# Standalone cascade POC. The Python package under src/ is not part of this build.
project(rci_detect LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Debug or Release" FORCE)
endif()

add_library(rci_detect STATIC src/detect.cpp)
target_include_directories(rci_detect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(rci_detect PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(rci_detect PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO)

add_executable(rci_detect_demo demos/demo.cpp)
target_link_libraries(rci_detect_demo PRIVATE rci_detect)
target_compile_options(rci_detect_demo PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(rci_detect_demo PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO)
