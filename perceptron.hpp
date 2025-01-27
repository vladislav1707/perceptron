#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <algorithm>

class NetConfig
{
public:
    static double GRADIENT_CLIP;
    static double BIAS;
    static double learningRate;
    static double LEAK_FACTOR;

    static void setParameters(double gradient_clip, double bias_value,
                              double learning_rate, double leak_factor)
    {
        GRADIENT_CLIP = gradient_clip; // gradient_clip
        BIAS = bias_value;             // bias
        learningRate = learning_rate;  // learning_rate
        LEAK_FACTOR = leak_factor;     // leak_factor
    }
};

//@ генерация случайных чисел
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

//@ Функция активации leaky ReLU
inline void lReLU(double &value)
{
    if (value < 0)
    {
        value = NetConfig::LEAK_FACTOR * value; // Коэффициент наклона 0.01 для отрицательных значений
    }
    // для положительных значений оставляем как есть
}

//@ Производная(deriviate) leaky ReLU
inline double lReLUDer(double value)
{
    if (value > 0)
    {
        return 1.0;
    }
    else
    {
        return NetConfig::LEAK_FACTOR; // Производная для отрицательной части
    }
}

//@ Функция для ограничения градиента
inline double clipGradient(double gradient)
{
    if (gradient > NetConfig::GRADIENT_CLIP)
        return NetConfig::GRADIENT_CLIP;
    if (gradient < -NetConfig::GRADIENT_CLIP)
        return -NetConfig::GRADIENT_CLIP;
    return gradient;
}

struct neuron
{
public:
    double value = 0.0;
    double error = 0.0;
    bool isBias = false;
};

class Network
{
public:
    int num_threads;
    int layers;
    neuron **neurons;
    double ***weights; // 1 слой, 2 номер нейрона, 3 связь с нейроном в следующем слое
    int *size;         // как topology но для Network

    Network(const std::vector<int> &topology, int threads = 4) : num_threads(threads)
    {
        omp_set_num_threads(num_threads);
        layers = topology.size();
        size = new int[layers];

        for (int i = 0; i < layers; i++)
        {
            if (i < layers - 1)
            {
                size[i] = topology[i] + 1; // +1 под bias в каждом скрытом слое
            }
            else
            {
                size[i] = topology[i]; // выходной слой без bias
            }
        }

        //@ создание нейронов
        neurons = new neuron *[layers];
        for (int i = 0; i < layers; i++)
        {
            neurons[i] = new neuron[size[i]];
            if (i < layers - 1) // Устанавливаем bias только для не выходных слоев
            {
                neurons[i][size[i] - 1].value = NetConfig::BIAS;
                neurons[i][size[i] - 1].isBias = true;
            }
        }

        //@ создание весов
        weights = new double **[layers - 1];
        for (int i = 0; i < layers - 1; i++)
        {
            weights[i] = new double *[size[i]];
            for (int j = 0; j < size[i]; j++)
            {
                weights[i][j] = new double[size[i + 1]];
                double scale = sqrt(2.0 / (size[i] * (1 + NetConfig::LEAK_FACTOR * NetConfig::LEAK_FACTOR)));
                // заполнение случайными значение от 0 до 1
                // Инициализация весов только не для bias нейронов
                if (!neurons[i][j].isBias)
                {
                    for (int k = 0; k < size[i + 1]; k++)
                    {
                        weights[i][j][k] = dis(gen) * scale; // реализация инициализации весов He
                    }
                }
                else
                {
                    // Для bias нейрона устанавливаем специальные веса
                    for (int k = 0; k < size[i + 1]; k++)
                    {
                        weights[i][j][k] = dis(gen) * 0.01; // Меньшие начальные веса для bias
                    }
                }
            }
        }
    }

    // метод для сохранения состояния сети в файл
    bool saveToFile(const std::string &filename)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "error: cant open file to save" << std::endl;
            return false;
        }

        // Сохраняем основные параметры сети
        file.write(reinterpret_cast<char *>(&layers), sizeof(layers));
        for (int i = 0; i < layers; i++)
        {
            file.write(reinterpret_cast<char *>(&size[i]), sizeof(size[i]));
        }

        // Сохраняем значения нейронов
        for (int i = 0; i < layers; i++)
        {
            for (int j = 0; j < size[i]; j++)
            {
                file.write(reinterpret_cast<char *>(&neurons[i][j].value), sizeof(double));
            }
        }

        // Сохраняем веса
        for (int i = 0; i < layers - 1; i++)
        {
            for (int j = 0; j < size[i]; j++)
            {
                for (int k = 0; k < size[i + 1]; k++)
                {
                    file.write(reinterpret_cast<char *>(&weights[i][j][k]), sizeof(double));
                }
            }
        }

        file.close();
        return true;
    }

    // метод для загрузки состояния сети из файла
    static Network *loadFromFile(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "error: cant open file to load" << std::endl;
            return nullptr;
        }

        // Читаем количество слоев
        int layers;
        file.read(reinterpret_cast<char *>(&layers), sizeof(layers));

        // Читаем размеры слоев и создаем топологию
        std::vector<int> topology(layers);
        for (int i = 0; i < layers; i++)
        {
            file.read(reinterpret_cast<char *>(&topology[i]), sizeof(int));
        }

        // Создаем новую сеть с загруженной топологией
        Network *network = new Network(topology);

        // Загружаем значения нейронов
        for (int i = 0; i < layers; i++)
        {
            for (int j = 0; j < topology[i]; j++)
            {
                file.read(reinterpret_cast<char *>(&network->neurons[i][j].value), sizeof(double));
            }
        }

        // Загружаем веса
        for (int i = 0; i < layers - 1; i++)
        {
            for (int j = 0; j < topology[i]; j++)
            {
                for (int k = 0; k < topology[i + 1]; k++)
                {
                    file.read(reinterpret_cast<char *>(&network->weights[i][j][k]), sizeof(double));
                }
            }
        }

        file.close();
        return network;
    }

    // Метод прямого распространения
    void forward(const std::vector<double> &input)
    {
#pragma omp parallel for
        // Установка входных значений (кроме bias)
        for (int i = 0; i < size[0] - 1; i++)
        {
            neurons[0][i].value = input[i];
        }

        // Проход по всем слоям, кроме входного
        for (int layer = 1; layer < layers; layer++)
        {
#pragma omp parallel for
            // Проход по всем нейронам текущего слоя
            for (int neuron = 0; neuron < size[layer]; neuron++)
            {
                if (!neurons[layer][neuron].isBias) // проверка чтобы нейрон не был bias
                {
                    double sum = 0;

                    // Вычисление взвешенной суммы
                    for (int prev = 0; prev < size[layer - 1]; prev++)
                    {
                        sum += neurons[layer - 1][prev].value * weights[layer - 1][prev][neuron];
                    }

                    // Применение функции активации
                    neurons[layer][neuron].value = sum;
                    lReLU(neurons[layer][neuron].value);
                }
            }
        }
    }

    // Метод обратного распространения ошибки
    void backprop(const std::vector<double> &target)
    {
        // 1. Вычисление ошибки для выходного слоя
        int output_layer = layers - 1;

#pragma omp parallel for
        for (int i = 0; i < size[output_layer]; i++)
        {
            double output = neurons[output_layer][i].value;
            // Производная квадратичной функции потерь
            // Обновление весов с ограничением градиента
            double error = (output - target[i]) * lReLUDer(output);
            neurons[output_layer][i].error = clipGradient(error);
        }

        // 2. Обратное распространение ошибки по скрытым слоям
        for (int layer = output_layer - 1; layer >= 0; layer--)
        {

            const int current_layer = layer;

#pragma omp parallel for
            for (int i = 0; i < size[current_layer]; i++)
            {
                double error = 0.0;
                // Суммируем взвешенные ошибки следующего слоя
                for (int j = 0; j < size[current_layer + 1]; j++)
                {
                    error += neurons[current_layer + 1][j].error * weights[current_layer][i][j];
                }
                neurons[current_layer][i].error = error * lReLUDer(neurons[current_layer][i].value);
            }
        }

        // 3. Обновление весов
        for (int layer = 0; layer < layers - 1; layer++)
        {
            const int current_layer = layer;

#pragma omp parallel for collapse(2)
            for (int i = 0; i < size[current_layer]; i++)
            {
                for (int j = 0; j < size[current_layer + 1]; j++)
                {
                    weights[current_layer][i][j] -= NetConfig::learningRate * neurons[current_layer][i].value * neurons[current_layer + 1][j].error;
                }
            }
        }
    }

    // --------------------------------------------------------
    //   реализация Q-обучения
    // --------------------------------------------------------
    // Предполагается, что сеть умеет выдавать Q-значения
    // во всех выходных нейронах (по одному на каждое действие).
    // Здесь:
    //   state      - текущее состояние среды (вектор входа)
    //   action     - индекс действия, которое мы применили
    //   reward     - полученная награда
    //   nextState  - новое состояние после действия
    //   done       - флаг конца эпизода
    //   gamma      - коэффициент дисконтирования
    void qLearn(const std::vector<double> &state,
                int action,
                double reward,
                const std::vector<double> &nextState,
                bool done,
                double gamma = 0.99)
    {
        // 1) Прямой проход (forward) для текущего состояния
        forward(state);
        std::vector<double> currentQ = getOutput();

        // 2) Если эпизод не завершен, моделируем следующее состояние
        double nextMax = 0.0;
        if (!done)
        {
            forward(nextState);
            std::vector<double> nextQ = getOutput();
            nextMax = *std::max_element(nextQ.begin(), nextQ.end());
        }

        // 3) Вычисляем целевое значение
        double updatedValue = reward + gamma * nextMax;

        // 4) Подготовливаем таргет (все выходы = текущие Q, кроме action)
        currentQ[action] = updatedValue;

        // 5) Вызываем уже готовый backprop с таргетом
        backprop(currentQ);
    }

    // метод управления потоками
    void setThreads(int threads)
    {
        num_threads = threads;
        omp_set_num_threads(num_threads);
    }

    double calculateError(const std::vector<double> &target)
    {
        double error = 0.0;
        int output_layer = layers - 1;
        for (int i = 0; i < size[output_layer]; i++)
        {
            double diff = neurons[output_layer][i].value - target[i];
            error += diff * diff;
        }
        return error / size[output_layer];
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

#endif