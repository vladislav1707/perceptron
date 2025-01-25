#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <fstream>
#include <cmath>
#include <omp.h> // Добавляем заголовок OpenMP

//@ генерация случайных чисел
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

//@ параметры
const double BIAS = 1.0;
double learningRate = 0.1;
std::vector<int> topology = {784, 256, 256, 10}; // нужно эксперементировать, вероятно мне нужно 512 нейронов в скрытых слоях

// Функция активации ReLU
inline void act(double &value)
{
    if (value < 0)
    {
        value = 0;
    }
    // иначе все остается как есть
}
// Производная ReLU
inline double actDerivative(double value)
{
    if (value > 0)
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
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

        // создание нейронов
        neurons = new neuron *[layers];
        for (int i = 0; i < layers; i++)
        {
            neurons[i] = new neuron[size[i]];
            if (i < layers - 1) // Устанавливаем bias только для не выходных слоев
            {
                neurons[i][size[i] - 1].value = BIAS;
                neurons[i][size[i] - 1].isBias = true;
            }
        }

        // создание весов
        weights = new double **[layers - 1];
        for (int i = 0; i < layers - 1; i++)
        {
            weights[i] = new double *[size[i]];
            for (int j = 0; j < size[i]; j++)
            {
                weights[i][j] = new double[size[i + 1]];
                // заполнение случайными значение от 0 до 1
                // Инициализация весов только не для bias нейронов
                if (!neurons[i][j].isBias)
                {
                    for (int k = 0; k < size[i + 1]; k++)
                    {
                        weights[i][j][k] = dis(gen) * sqrt(2.0 / size[i]); // реализация инициализации весов He
                    }
                }
                else
                {
                    // Для bias нейрона устанавливаем специальные веса
                    for (int k = 0; k < size[i + 1]; k++)
                    {
                        weights[i][j][k] = dis(gen) * 0.1; // Меньшие начальные веса для bias
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

    // Новый метод для загрузки состояния сети из файла
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
                    act(neurons[layer][neuron].value);
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
            neurons[output_layer][i].error = (output - target[i]) * actDerivative(output);
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
                neurons[current_layer][i].error = error * actDerivative(neurons[current_layer][i].value);
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
                    weights[current_layer][i][j] -= learningRate * neurons[current_layer][i].value * neurons[current_layer + 1][j].error;
                }
            }
        }
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