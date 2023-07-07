//
// Created by 1flor on 03/06/2023.
//

#ifndef NEURALNETWORK_GUI_H
#define NEURALNETWORK_GUI_H

#include <GLFW/glfw3.h>
#include <thread>

class GUIWindow {
private:
    GLFWwindow* window;
    int width;
    int height;
    const char* title;
    std::thread renderThread;

    //Initializes the GUIWindow
    void initialize();

    //Handles the rendering of the GUI client
    void render();

    //Cleans up and shuts down the GUI window
    void shutdown();

public:
    GUIWindow(int width, int height, const char* title);
    ~GUIWindow();

    //Starts the thread on which the rendering loop is executed
    void run();

    //Used to wait for the rendering thread to finish
    void join();
};

#endif //NEURALNETWORK_GUI_H
