#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include "src/headers/NeuralNetwork.h"

std::vector<std::vector<double>> extractData(const std::string &datasetPath) {
    std::ifstream file(datasetPath);
    if (!file.is_open()) {
        std::cout << "Couldn't open file" << std::endl;
        exit(EXIT_FAILURE);
    }

    //The expected output for any line of data is placed at data[n][0]
    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::vector<double> dataLine;
        std::istringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            //dataLine.push_back(std::stoi(value) / 255.0);
            dataLine.push_back(std::stoi(value));
        }
        data.push_back(dataLine);
    }
    return data;
}

std::vector<std::vector<double>> extractExpectedOutputs(std::vector<std::vector<double>> &data, int outputsSize) {
    std::vector<std::vector<double>> expectedOutputs;

    for (auto &dataLine: data) {
        //Initialize all values as 0 and set the correct node to activation 1
        std::vector<double> expectedOutput(outputsSize, 0);
        expectedOutput[dataLine[0]] = 1;

        //Add expectedOutput and erase the expected output from the dataset
        expectedOutputs.push_back(expectedOutput);
        dataLine.erase(dataLine.begin());
    }

    return expectedOutputs;
}

template <typename T>
std::vector<T> getRandomSubset(std::vector<T> largerVector, int subsetSize) {
    // Create an empty vector to store the selected values
    std::vector<T> subset;

    // Create a random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> dist(0, largerVector.size() - subsetSize);

    int startingIndex = dist(generator);
    subsetSize = startingIndex + subsetSize;

    // Copy the randomly selected values to the subset vector
    for (int i = startingIndex; i < subsetSize; ++i) {
        subset.push_back(largerVector[i]);
    }

    return subset;
}

int main() {
    std::string datasetPath = R"(C:\Users\1flor\CLionProjects\NeuralNetwork\src\dataset\fruits\fruits_train.csv)";
    std::cout << "Processing data..." << std::endl;
    std::vector<std::vector<double>> data = extractData(datasetPath);
    std::cout << "Creating expected outputs..." << std::endl;
    std::vector<std::vector<double>> expectedOutputs = extractExpectedOutputs(data, 2);
    std::cout << "Done" << std::endl;

    //Create and populate the data points
    std::vector<neuralNet::DataPoint> dataPoints;
    for (int index = 0; index < data.size(); index++) {
        dataPoints.emplace_back(data[index], expectedOutputs[index]);
    }

    //int layerSizes[4] = {784, 30, 20, 10};
    int layerSizes[4] = {2, 3, 3, 2};
    neuralNet::NeuralNetwork neuralNetwork(layerSizes, 4);

    std::vector<neuralNet::DataPoint> dataPoint = getRandomSubset(dataPoints, 50);
    std::cout << "Initial cost: " << neuralNetwork.cost(dataPoint) << std::endl;

    for (int iteration = 0; iteration < 10000; iteration++) {
        neuralNetwork.gradientDescent(dataPoint);
        std::cout << "Cost: " <<  neuralNetwork.cost(dataPoint) << std::endl;
        dataPoint = getRandomSubset(dataPoints, 50);
    }

    std::cout << "Cost: " <<  neuralNetwork.cost(dataPoint) << std::endl;

    return 0;
}
