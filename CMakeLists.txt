cmake_minimum_required (VERSION 2.6)

project (OrbitalSpace)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "./cmake/")

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -D_DEBUG")

find_package(Threads REQUIRED)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_MIN_VERSION "1.46.0")
find_package(Boost COMPONENTS thread program_options system date_time REQUIRED)

find_package(OpenGL REQUIRED) # includes GLU

find_package(GLUT REQUIRED)

# TODO? find_package(GLEW REQUIRED)

# set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 2 COMPONENTS audio graphics network system window REQUIRED)

# TODO? find_package(Eigen3 REQUIRED)

# No include for Threads
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenGL_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${SFML_INCLUDE_DIRS})
# TODO what?? Fix plz
include_directories("/usr/include/eigen3")
include_directories("/usr/include/eigen3/unsupported")
include_directories("./include")

add_subdirectory("./SDL")

add_definitions(-DGL_GLEXT_PROTOTYPES)
add_definitions(-D__STDC_LIMIT_MACROS)

add_executable(OrbitalSpace
  src/main.cpp
  src/orApp.cpp
  src/orCamera.cpp
  src/orEntity.cpp
  src/orPhysics.cpp
  src/orRender.cpp
  src/util.cpp
  src/orProfile/perftimer.cpp
  src/timer.cpp
)
target_link_libraries(OrbitalSpace ${CMAKE_THREAD_LIBS_INIT}) # special case because pthread isn't always called pthread
target_link_libraries(OrbitalSpace ${Boost_LIBRARIES})
target_link_libraries(OrbitalSpace ${OpenGL_LIBRARIES})
target_link_libraries(OrbitalSpace GLU)
target_link_libraries(OrbitalSpace ${GLUT_LIBRARIES})
target_link_libraries(OrbitalSpace GLEW) # TODO # TODO...what?
target_link_libraries(OrbitalSpace ${SFML_LIBRARIES})
target_link_libraries(OrbitalSpace SDL2)
