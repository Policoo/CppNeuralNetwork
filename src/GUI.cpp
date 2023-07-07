//
// Created by 1flor on 03/06/2023.
//

#include "headers/GUI.h"
#include <iostream>
#include "../libs/imgui/imgui.h"
#include "../libs/imgui/imgui_impl_opengl3.h"
#include "../libs/imgui/imgui_impl_glfw.h"

GUIWindow::GUIWindow(int width, int height, const char *title) {
    this->width = width;
    this->height = height;
    this->title = title;

    initialize();
}

GUIWindow::~GUIWindow() {
    shutdown();
}

void GUIWindow::run() {
    while (!glfwWindowShouldClose(window)) {
        render();

        //A frame is rendered in the background, swapping buffers and poll events moves it, so it is visible
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void GUIWindow::join() {
    if (renderThread.joinable()) {
        renderThread.join();
    }
}

void GUIWindow::initialize() {
    if (!glfwInit()) {
        std::cout << "Could not initialize GUI window. glfwInit() Failed" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    //These window hints ensures the window uses OpenGL v3.3 and indicate that we want to use the core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    //Create the window
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cout << "Could not initialize GUI window. glfwCreateWindow() Failed" << std::endl;
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();
}

void GUIWindow::render() {
    //Start a new ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    //Add ImGui UI elements and logic

    //Example: Create a window with a button that changes the window's background color
    if (ImGui::Begin("My Window")) {
        if (ImGui::Button("Change Color")) {
            // Change the window's background color
            glClearColor(1.0f, 1.0f, 0.0f, 1.0f);
        }
    }
    ImGui::End();

    //Render ImGui UI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    //Restore the default OpenGL clear color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void GUIWindow::shutdown() {
    join();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}
