cmake_minimum_required (VERSION 2.6)

project (OrbitalSpace)

#set(CMAKE_CXX_FLAGS "-g3 -Wall -Wextra -Wno-shadow -Wno-unused-function")
#set(CMAKE_CXX_FLAGS "-g3 -Wall -O3")
set(CMAKE_CXX_FLAGS "-g3 -Wall -D_DEBUG")
set(CMAKE_EXE_LINKER_FLAGS "-L/media/Data2/Coding/SFML-2.0-rc/lib")

include_directories(${Boost_INCLUDE_DIRS})
include_directories("/media/Data2/Coding/SFML-2.0-rc/include")
include_directories("/usr/include/eigen3")
include_directories("/usr/include/eigen3/unsupported")
include_directories("include")

find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_MIN_VERSION "1.46.0")
find_package(Boost COMPONENTS thread program_options system date_time REQUIRED)

# find_package(SFML REQUIRED)

add_definitions(-DGL_GLEXT_PROTOTYPES)
add_definitions(-D__STDC_LIMIT_MACROS)

add_executable(OrbitalSpace
  src/main.cpp
  src/app.cpp
  src/orbitalSpaceApp.cpp
  src/orCamera.cpp
  src/orEntity.cpp
  src/orPhysics.cpp
  src/orRender.cpp
  src/util.cpp
  src/orProfile/perftimer.cpp
  src/timer.cpp
)
target_link_libraries(OrbitalSpace GL GLU GLEW glut sfml-graphics sfml-audio sfml-window sfml-system)
target_link_libraries(OrbitalSpace ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
