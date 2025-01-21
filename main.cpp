#include <iostream>
#include <random>
#include <chrono>

//@ random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

//@ parameters
const double BIAS = 1.0;
std::vector<int> topology = {784, 256, 256, 10}; // need experiments, may i need set 512 neurons in hidden layers

void act(double &value) // using ReLU
{
    if (value < 0)
    {
        value = 0;
    }
    // otherwise everything is left as is
}

struct neuron
{
public:
    double value = round(dis(gen) * 100.0) / 100.0;
    double error;
};

class Network
{
public:
    int layers;
    neuron **neurons;
    double ***weights; // 1 layer, 2 neuron number, 3 connection number of the neuron in the next layer
    int *size;

    Network(const std::vector<int> &topology)
    {
        // initialization of layers count and size array that stores neurons count for each layer
        layers = topology.size();

        size = new int[layers];
        for (int i = 0; i < layers; i++)
        {
            size[i] = topology[i];
        }

        // creating neurons
        neurons = new neuron *[layers];
        for (int i = 0; i < layers; i++)
        {
            neurons[i] = new neuron[topology[i]];
        }

        // creating weights
        weights = new double **[layers - 1];
        for (int i = 0; i < layers - 1; i++)
        {
            weights[i] = new double *[topology[i]];
            for (int j = 0; j < topology[i]; j++)
            {
                weights[i][j] = new double[topology[i + 1]];
                // fill with random from 0 to 1
                for (int k = 0; k < topology[i + 1]; k++)
                {
                    weights[i][j][k] = dis(gen);
                }
            }
        }
    }
    ~Network()
    {
        // Freeing up memory for weights
        for (int i = 0; i < layers - 1; i++)
        {
            for (int j = 0; j < size[i]; j++)
            {
                delete[] weights[i][j];
            }
            delete[] weights[i];
        }
        delete[] weights;

        // Freeing up memory for neurons
        for (int i = 0; i < layers; i++)
        {
            delete[] neurons[i];
        }
        delete[] neurons;

        // Free up memory for the size array
        delete[] size;
    }
};

int main(int argc, char **args)
{
    Network net(topology);
}