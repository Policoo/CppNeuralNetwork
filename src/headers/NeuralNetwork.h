//
// Created by 1flor on 28/05/2023.
//

#ifndef UNTITLED1_NEURALNETWORK_H
#define UNTITLED1_NEURALNETWORK_H

#include <vector>

namespace neuralNet {
    class DataPoint {
    private:
        std::vector<double> inputData;
        std::vector<double> expectedOutputs;

    public:
        DataPoint(std::vector<double> inputData, std::vector<double> expectedOutputs);

        std::vector<double> getInputData() {
            return inputData;
        }

        std::vector<double> getExpectedOutputs() {
            return expectedOutputs;
        }

        void print();
    };

    class Layer {
    private:
        int numNodesIn;
        int numNodesOut;

        //Weights of the connections between the last layer and this one
        std::vector<std::vector<double>> weights;

        //Biases for all the nodes of this layer, these acts as a sort of activation threshold
        std::vector<double> biases;

        /* Stores the activation values of each node. These starts as 0 and they
         fill up as the network processes information */
        std::vector<double> activations;

        //Stores the last inputs it received. Used for calculating the derivative cost/weight
        std::vector<double> inputs;

        //These store the gradient of the cost for a given weight or bias
        std::vector<std::vector<double>> costGradientW;
        std::vector<double> costGradientB;

        //Assigns random
        void randomizeWeightsAndBiases();

        //Applies a sigmoid function to the activation value of a node
        double activationSigmoid(double input);

        //Calculates the derivative of the sigmoid function, with respect to the weighted input
        double activationSigmoidDerivative(double input);

        //Calculates the derivative of the cost, with respect to the activation value
        double calculateCostDerivative(double outputActivation, double expectedOutput);

    public:
        //Default constructor for layer
        Layer() = default;

        //Initializes a layer with the number of incoming nodes and outgoing nodes
        Layer(int numNodesIn, int numNodesOut);

        //Returns the number of nodes in this layer
        int length() const;

        //Returns the number of incoming nodes
        int nodesIn() const;

        //Returns the activation numbers
        std::vector<double> getActivations();

        //Adjusts the weight of a connection by adding the value
        void adjustWeight(int nodeIn, int nodeOut, double value);

        //Adjusts a node's bias by adding the value
        void adjustBias(int node, double value);

        //Sets the cost gradient for a connection to the specified value
        void setCostGradientW(int nodeIn, int nodeOut, double value);

        //Sets the cost gradient for a node's bias to the specified value
        void setCostGradientB(int node, double value);

        //Calculates the outputs (values of all the nodes of this layer)
        void calculateOutputs(std::vector<double> inputs);

        //Calculates the cost of a node
        double calculateCost(double outputActivation, double expectedOutput);

        //Applies all the gradients stored in the costGradient vectors
        void applyGradients(double learnRate);

        //Calculates the cost gradients based on the gradient product
        void calculateGradients(std::vector<double> gradientProducts);

        //Calculates the gradient product for the nodes in the output layer
        std::vector<double> outputLayerGradientProduct(std::vector<double> expectedOutputs);

        //Calculates the gradient product for the nodes in a hidden layer
        std::vector<double> hiddenLayerGradientProduct(Layer oldLayer, std::vector<double> oldGradientProducts);

        void printNodes();
    };

    class NeuralNetwork {
    private:
        std::vector<Layer> layers;

        //Calculates the outputs of all layers
        std::vector<double> calculateOutputs(std::vector<double> inputs);

        //Calculates the cost for a set of inputs
        double calculateCost(std::vector<double> inputs, std::vector<double> expectedOutputs);

        //Returns the output layer of the network
        Layer outputLayer();

        //Applies the cost gradients to all the layers in the network
        void applyAllGradients(double learnRate);

        /* Back propagates through the network, adjusting the weights and biases
        based on the gradient product of the layer after it */
        void backPropagation(std::vector<double> inputs, std::vector<double> expectedOutputs);

    public:
        //Initializes the neural network with the specified number of layers
        NeuralNetwork(std::vector<int> layersInfo);

        //Gets the output node with the highest activation value
        int classify(std::vector<double> inputs);

        //Calculates the average cost over all inputs
        double cost(std::vector<DataPoint> dataPoints);

        //Makes the neural network gradientDescent, based on the inputs and the expected outputs
        void gradientDescent(std::vector<DataPoint> dataPoints);
    };
}

#endif //UNTITLED1_NEURALNETWORK_H