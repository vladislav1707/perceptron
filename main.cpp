#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>
#include "perceptron.hpp"

const std::vector<int> topology = {784, 128, 64, 10};

// Функция для определения предсказанной цифры
int getPredictedDigit(const std::vector<double> &output)
{
    return std::max_element(output.begin(), output.end()) - output.begin();
}

// Функция для определения правильной цифры из one-hot encoding
int getActualDigit(const std::vector<double> &label)
{
    return std::max_element(label.begin(), label.end()) - label.begin();
}
// Функция для вычисления точности
double calculateAccuracy(int correct, int total)
{
    if (total <= 0)
        return 0.0;
    return (static_cast<double>(correct) / total) * 100.0;
}

// Функция для чтения MNIST изображений
std::vector<std::vector<double>> readMNISTImages(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    // Чтение заголовка
    file.read((char *)&magic_number, sizeof(magic_number));
    file.read((char *)&number_of_images, sizeof(number_of_images));
    file.read((char *)&rows, sizeof(rows));
    file.read((char *)&cols, sizeof(cols));

    // Преобразование из big-endian в little-endian
    magic_number = __builtin_bswap32(magic_number);
    number_of_images = __builtin_bswap32(number_of_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    // Чтение данных изображений
    std::vector<std::vector<double>> images(number_of_images, std::vector<double>(rows * cols));
    for (int i = 0; i < number_of_images; i++)
    {
        for (int j = 0; j < rows * cols; j++)
        {
            unsigned char pixel = 0;
            file.read((char *)&pixel, sizeof(pixel));
            // Нормализация значений пикселей до диапазона [0, 1]
            images[i][j] = static_cast<double>(pixel) / 255.0;
        }
    }

    return images;
}

// Функция для чтения MNIST меток
std::vector<std::vector<double>> readMNISTLabels(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int magic_number = 0;
    int number_of_labels = 0;

    // Чтение заголовка
    file.read((char *)&magic_number, sizeof(magic_number));
    file.read((char *)&number_of_labels, sizeof(number_of_labels));

    // Преобразование из big-endian в little-endian
    magic_number = __builtin_bswap32(magic_number);
    number_of_labels = __builtin_bswap32(number_of_labels);

    if (magic_number != 2049)
    {
        throw std::runtime_error("Invalid MNIST image file format");
    }

    // Чтение меток и преобразование в one-hot encoding
    std::vector<std::vector<double>> labels(number_of_labels, std::vector<double>(10, 0.0));
    for (int i = 0; i < number_of_labels; i++)
    {
        unsigned char label = 0;
        file.read((char *)&label, sizeof(label));
        labels[i][label] = 1.0;
    }

    return labels;
}

int main(int argc, char **args)
{
    int train;
    int saveModel;
    int loadModel;
    int testMode;
    std::string whereToSave;
    std::string whereToLoad;

    std::cout << "provide training?(1 yes, 0 no): ";
    std::cin >> train;
    if (train == 1)
    {
        std::cout << "save new model?(1 yes, 0 no): ";
        std::cin >> saveModel;
    }
    if (saveModel == 1)
    {
        std::cout << "where to save?(filename): ";
        std::cin >> whereToSave;
    }
    std::cout << "load model?(1 yes, 0 no): ";
    std::cin >> loadModel;
    if (loadModel == 1)
    {
        std::cout << "where to load?(filename): "; // откуда загружать данные в нейросеть(имя файла)
        std::cin >> whereToLoad;
    }
    std::cout << "test model?(1 yes, 0 no): ";
    std::cin >> testMode;

    try
    {
        // Загрузка обучающих данных
        std::vector<std::vector<double>> train_images;
        std::vector<std::vector<double>> train_labels;

        if (testMode == 1)
        {
            train_images = readMNISTImages("t10k-images.idx3-ubyte");
            train_labels = readMNISTLabels("t10k-labels.idx1-ubyte");
        }
        else
        {
            train_images = readMNISTImages("train-images.idx3-ubyte");
            train_labels = readMNISTLabels("train-labels.idx1-ubyte");
        }

        std::cout << "Loaded " << train_images.size() << " training images" << std::endl;

        Network *net;
        if (loadModel == 1)
        {
            // Загружаем сеть из файла
            net = Network::loadFromFile(whereToLoad);
            if (net == nullptr)
            {
                throw std::runtime_error("ERROR: cant load model");
            }
        }
        else
        {
            // Создаем новую сеть
            net = new Network(topology, 4);
        }

        int correctPredictions = 0;
        int totalPredictions = 0;
        double totalError = 0.0;
        double avgError = 0.0;

        // Используем сеть
        for (size_t i = 0; i < train_images.size(); ++i)
        {
            totalPredictions++;
            net->forward(train_images[i]);

            double currentError = net->calculateError(train_labels[i]);
            totalError += currentError;
            avgError = totalError / totalPredictions;

            if (train == 1)
            {
                net->backprop(train_labels[i]);
            }

            if (getPredictedDigit(net->getOutput()) == getActualDigit(train_labels[i]))
            {
                correctPredictions++;
            }

            if (i % 1000 == 0 || i == train_images.size() - 1)
            {
                double accuracy = calculateAccuracy(correctPredictions, totalPredictions);
                std::cout << "\n\nProcessed " << (i) << " images."
                          << "\nCurrent accuracy: " << accuracy << "%"
                          << "\nAverage error: " << avgError;
            }
        }

        if (saveModel == 1)
        {
            net->saveToFile(whereToSave);
        }

        // Освобождаем память
        delete net;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        int exit;
        std::cout << "\n"
                  << "write 1 to exit" << std::endl;
        std::cin >> exit;
        if (exit == 1)
        {
            return 1;
        }
    }

    int exit;
    std::cout << "\n"
              << "write 1 to exit" << "\n";
    std::cin >> exit;
    if (exit == 1)
    {
        return 1;
    }
    return 0;
}