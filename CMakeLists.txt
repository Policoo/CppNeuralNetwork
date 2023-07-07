cmake_minimum_required(VERSION 3.23)
project(CppNeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

# Set the path to the GLFW library
set(GLFW_DIR /mingw64)

find_package(glfw3 REQUIRED)

include_directories(${GLFW_INCLUDE_DIRS})
include_directories(libs)

# Find and link the OpenGL library
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

add_executable(CppNeuralNetwork main.cpp src/headers/NeuralNetwork.h src/NeuralNetwork.cpp src/test.cpp
        src/headers/GUI.h src/GUI.cpp libs/imgui/imgui.cpp libs/imgui/imgui_draw.cpp libs/imgui/imgui_tables.cpp libs/imgui/imgui_widgets.cpp
        libs/imgui/imgui_impl_glfw.cpp libs/imgui/imgui_impl_opengl3.cpp libs/imgui/imgui_demo.cpp)

# Link the GLFW and OpenGL libraries with your project
target_link_libraries(CppNeuralNetwork PRIVATE glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES})