//
// Created by 1flor on 28/05/2023.
//

#include <limits>
#include <cmath>
#include "headers/NeuralNetwork.h"
#include <iostream>
#include <random>
#include <utility>

using namespace neuralNet;

// <-- LAYER IMPLEMENTATION --> //

Layer::Layer(int numNodesIn, int numNodesOut) {
    this->numNodesIn = numNodesIn;
    this->numNodesOut = numNodesOut;

    //Initialize activations and set all the values to 0
    activations.resize(numNodesOut, 0);

    inputs.resize(numNodesIn);

    //Initialize all the weights between the previous layer and this one
    weights.resize(numNodesIn, std::vector<double>(numNodesOut));
    costGradientW.resize(numNodesIn, std::vector<double>(numNodesOut));

    //Initialize a bias for each node
    biases.resize(numNodesOut);
    costGradientB.resize(numNodesOut);

    randomizeWeightsAndBiases();
}

void Layer::randomizeWeightsAndBiases() {
    //Generate random numbers based on the Gaussian distribution
    std::random_device random;
    std::mt19937 gen(random());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            weights[nodeIn][nodeOut] = distribution(gen);
        }
    }

    for (int node = 0; node < numNodesOut; node++) {
        biases[node] = distribution(gen);
    }
}

void Layer::calculateOutputs(std::vector<double> inputs) {
    this->inputs = inputs;

    //For each node in this layer
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
        //Add the bias of this node
        double weightedInput = biases[nodeOut];

        /* For each node in the previous layer, multiply its activation value by the weight
           of the connection between this node and the one from the previous layer.
           Formula ends up being: b + a1 * w1 + a2 * w2 + ... */
        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
            weightedInput += inputs[nodeIn] * weights[nodeIn][nodeOut];
        }

        //Apply the activation function and add the activation value of this node
        activations[nodeOut] = activationSigmoid(weightedInput);
    }
}

double Layer::activationSigmoid(const double input) {
    return 1.0 / (1.0 + exp(-input));
}

double Layer::activationSigmoidDerivative(double input) {
    double activation = activationSigmoid(input);
    return activation * (1 - activation);
}

double Layer::calculateCost(double outputActivation, double expectedOutput) {
    double error = outputActivation - expectedOutput;
    return error * error;
}

double Layer::calculateCostDerivative(double outputActivation, double expectedOutput) {
    return 2 * (outputActivation - expectedOutput);
}

int Layer::length() const {
    return numNodesOut;
}

int Layer::nodesIn() const {
    return numNodesIn;
}

std::vector<double> Layer::getActivations() {
    return activations;
}

void Layer::adjustWeight(int nodeIn, int nodeOut, double value) {
    weights[nodeIn][nodeOut] += value;
}

void Layer::adjustBias(int node, double value) {
    biases[node] += value;
}

void Layer::setCostGradientW(int nodeIn, int nodeOut, double value) {
    costGradientW[nodeIn][nodeOut] = value;
}

void Layer::setCostGradientB(int node, double value) {
    costGradientB[node] = value;
}

void Layer::printNodes() {
    for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            std::cout << "Node in: " << nodeIn << ", Node out: " << nodeOut << ", Value: " << weights[nodeIn][nodeOut]
                      << std::endl;
        }
    }
}

void Layer::applyGradients(double learnRate) {
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
        biases[nodeOut] -= costGradientB[nodeOut] * learnRate;

        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
            weights[nodeIn][nodeOut] -= costGradientW[nodeIn][nodeOut] * learnRate;
        }
    }
}

std::vector<double> Layer::outputLayerGradientProduct(std::vector<double> expectedOutputs) {
    std::vector<double> gradientProducts(length());

    for (int node = 0; node < length(); node++) {
        //Evaluate partial derivatives for current node: cost/activation * activation/weightedInput
        gradientProducts[node] = activationSigmoidDerivative(activations[node])
                                 * calculateCostDerivative(activations[node], expectedOutputs[node]);
    }

    return gradientProducts;
}

std::vector<double> Layer::hiddenLayerGradientProduct(Layer oldLayer, std::vector<double> oldGradientProducts) {
    std::vector<double> gradientProducts(length());

    for (int newGradientIndex = 0; newGradientIndex < gradientProducts.size(); newGradientIndex++) {
        double gradientProductValue = 0;

        for (int oldGradientIndex = 0; oldGradientIndex < oldGradientProducts.size(); oldGradientIndex++) {
            //Partial derivative of the weighted input with respect to the input
            gradientProductValue += oldLayer.weights[newGradientIndex][oldGradientIndex]
                                    * oldGradientProducts[oldGradientIndex];
        }

        gradientProducts[newGradientIndex] = activationSigmoidDerivative(gradientProductValue);
    }

    return gradientProducts;
}

void Layer::calculateGradients(std::vector<double> gradientProducts) {
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
        for (int nodeIn = 0; nodeIn < numNodesIn; ++nodeIn) {
            costGradientW[nodeIn][nodeOut] += inputs[nodeIn] * gradientProducts[nodeOut];
        }

        costGradientB[nodeOut] = gradientProducts[nodeOut];
    }
}

// <-- LAYER IMPLEMENTATION END --> //

// <-- NEURAL NETWORK IMPLEMENTATION --> //

NeuralNetwork::NeuralNetwork(int *layerSizes, int numOfLayers) {
    layers.resize(numOfLayers - 1);
    for (int index = 0; index < numOfLayers - 1; index++) {
        layers[index] = Layer(layerSizes[index], layerSizes[index + 1]);
    }
}

std::vector<double> NeuralNetwork::calculateOutputs(std::vector<double> inputs) {
    //Give the first layer the inputs
    layers[0].calculateOutputs(inputs);

    //Calculate the output of each layer and feed it as an input to the next layer
    for (int layer = 1; layer < layers.size(); layer++) {
        layers[layer].calculateOutputs(layers[layer - 1].getActivations());
    }
    return outputLayer().getActivations();
}

int NeuralNetwork::classify(std::vector<double> inputs) {
    calculateOutputs(std::move(inputs));
    double maxValue = std::numeric_limits<double>::lowest();
    int maxNode = 0;

    //Go through the output nodes and find the one with the highest activation value
    for (int node = 0; node < outputLayer().length(); node++) {
        if (outputLayer().getActivations()[node] > maxValue) {
            maxValue = outputLayer().getActivations()[node];
            maxNode = node;
        }
    }
    return maxNode;
}

double NeuralNetwork::calculateCost(std::vector<double> inputs, std::vector<double> expectedOutputs) {
    std::vector<double> outputs = calculateOutputs(std::move(inputs));
    double cost = 0;

    //Add up the cost from each of the outputs
    for (int nodeOut = 0; nodeOut < outputLayer().length(); nodeOut++) {
        cost += outputLayer().calculateCost(outputs[nodeOut], expectedOutputs[nodeOut]);
    }
    return cost;
}

double NeuralNetwork::cost(std::vector<DataPoint> dataPoints) {
    double totalCost = 0;
    int datasetSize = dataPoints.size();

    //Add up the costs from all data points
    for (auto &dataPoint: dataPoints) {
        totalCost += calculateCost(dataPoint.getInputData(), dataPoint.getExpectedOutputs());
    }

    //Return the average cost between the data points
    return totalCost / datasetSize;
}

Layer NeuralNetwork::outputLayer() {
    return layers[layers.size() - 1];
}

void NeuralNetwork::gradientDescent(std::vector<DataPoint> dataPoints) {
    double learnRate = 0.1;

    for (auto &dataPoint: dataPoints) {
        backPropagation(dataPoint.getInputData(), dataPoint.getExpectedOutputs());
    }

    applyAllGradients(learnRate / dataPoints.size());
}

void NeuralNetwork::applyAllGradients(double learnRate) {
    for (auto &layer: layers) {
        layer.applyGradients(learnRate);
    }
}

void NeuralNetwork::backPropagation(std::vector<double> inputs, std::vector<double> expectedOutputs) {
    //Run the inputs through the network
    calculateOutputs(std::move(inputs));

    //Update the gradients of the output layer
    std::vector<double> gradientProducts = outputLayer().outputLayerGradientProduct(std::move(expectedOutputs));
    outputLayer().calculateGradients(gradientProducts);

    //Calculate the gradients for each of the hidden layers
    for (int layer = layers.size() - 2; layer >= 0; layer--) {
        gradientProducts = layers[layer].hiddenLayerGradientProduct(layers[layer + 1], gradientProducts);
        layers[layer].calculateGradients(gradientProducts);
    }
}


// <-- NEURAL NETWORK IMPLEMENTATION END --> //

DataPoint::DataPoint(std::vector<double> inputData, std::vector<double> expectedOutputs) {
    this->inputData = std::move(inputData);
    this->expectedOutputs = std::move(expectedOutputs);
}

void DataPoint::print() {
    std::cout << "Inputs: ";
    for (auto &input : inputData) {
        std::cout << input << " ";
    }
    std::cout << " -> Expected outputs: ";
    for (auto &expectedOutput : expectedOutputs) {
        std::cout << expectedOutput << " ";
    }
    std::cout << std::endl;
}
