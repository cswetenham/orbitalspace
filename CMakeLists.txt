cmake_minimum_required (VERSION 2.6)

project (OrbitalSpace)

set(CMAKE_CXX_FLAGS "-g3 -Wall -Wextra -Wno-shadow")

include_directories("/media/Data2/Coding/SFML-2.0-rc/include")
include_directories("/usr/include/eigen3")
include_directories("/usr/include/eigen3/unsupported")
include_directories("include")

find_package(OpenGL)
find_package(Threads)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_MIN_VERSION "1.46.0")
find_package(Boost COMPONENTS thread program_options system REQUIRED)

add_definitions(-DGL_GLEXT_PROTOTYPES)
add_definitions(-D__STDC_LIMIT_MACROS)

add_executable(OrbitalSpace
  src/main.cpp
  src/orbitalSpaceApp.cpp
  src/app.cpp
  src/taskSchedulerWorkStealing.cpp
  src/util.cpp
  src/perftimer.cpp
  src/timer.cpp
)
target_link_libraries(OrbitalSpace GL GLU glut)
