#include <iostream>
#include <random>
#include <chrono>
#include <vector>

//@ random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

//@ parameters
const double BIAS = 1.0;
double learningRate;
std::vector<int> topology = {784, 256, 256, 10}; // нужно эксперементировать, вероятно мне нужно 512 нейронов в скрытых слоях

void act(double &value) // тут используется функция активации ReLU
{
    if (value < 0)
    {
        value = 0;
    }
    // иначе все остается как есть
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
    double ***weights; // 1 слой, 2 номер нейрона, 3 связь с нейроном в следующем слое
    int *size;

    Network(const std::vector<int> &topology)
    {
        // инициализация количества слоев и массива size, который хранит количество нейронов для каждого слоя
        layers = topology.size();

        size = new int[layers];
        for (int i = 0; i < layers; i++)
        {
            size[i] = topology[i];
        }

        // создание нейронов
        neurons = new neuron *[layers];
        for (int i = 0; i < layers; i++)
        {
            neurons[i] = new neuron[topology[i]];
        }

        // создание весов
        weights = new double **[layers - 1];
        for (int i = 0; i < layers - 1; i++)
        {
            weights[i] = new double *[topology[i]];
            for (int j = 0; j < topology[i]; j++)
            {
                weights[i][j] = new double[topology[i + 1]];
                // заполнение случайными значение от 0 до 1
                for (int k = 0; k < topology[i + 1]; k++)
                {
                    weights[i][j][k] = dis(gen);
                }
            }
        }
    }
    // Метод прямого распространения
    void forward(const std::vector<double> &input)
    {
        // Установка входных значений
        for (int i = 0; i < size[0]; i++)
        {
            neurons[0][i].value = input[i];
        }

        // Проход по всем слоям, кроме входного
        for (int layer = 1; layer < layers; layer++)
        {
            // Проход по всем нейронам текущего слоя
            for (int neuron = 0; neuron < size[layer]; neuron++)
            {
                double sum = 0;

                // Вычисление взвешенной суммы
                for (int prev = 0; prev < size[layer - 1]; prev++)
                {
                    sum += neurons[layer - 1][prev].value * weights[layer - 1][prev][neuron];
                }

                // Добавление bias
                sum += BIAS;

                // Применение функции активации
                neurons[layer][neuron].value = sum;
                act(neurons[layer][neuron].value);
            }
        }
    }

    // Получение выходного слоя(получение вероятностей ответа нейросети)
    std::vector<double> getOutput()
    {
        std::vector<double> output(size[layers - 1]);
        for (int i = 0; i < size[layers - 1]; i++)
        {
            output[i] = neurons[layers - 1][i].value;
        }
        return output;
    }
    ~Network()
    {
        // освобождение памяти от весов
        for (int i = 0; i < layers - 1; i++)
        {
            for (int j = 0; j < size[i]; j++)
            {
                delete[] weights[i][j];
            }
            delete[] weights[i];
        }
        delete[] weights;

        // освобождение памяти от нейронов
        for (int i = 0; i < layers; i++)
        {
            delete[] neurons[i];
        }
        delete[] neurons;

        // освобождение памяти для массива размеров
        delete[] size;
    }
};

int main(int argc, char **args)
{
    std::cout << "Enter learning rate (recommended range 0.0001 - 0.1): ";
    std::cin >> learningRate;
    Network net(topology);
}