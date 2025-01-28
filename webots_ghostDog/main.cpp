#include <webots/Robot.hpp>
#include <webots/Motor.hpp>
#include <webots/DistanceSensor.hpp>
#include <webots/Gyro.hpp>
#include <webots/PositionSensor.hpp>
#include "../perceptron.hpp"

double NetConfig::GRADIENT_CLIP = 1.0;
double NetConfig::BIAS = 1.0;
double NetConfig::learningRate = 0.01;
double NetConfig::LEAK_FACTOR = 0.01;

// Топология сети для GhostDog:
// Входной слой (24):
// - 8 значений с датчиков расстояния
// - 6 значений с гироскопа (ориентация)
// - 10 значений с датчиков положения суставов
// Скрытые слои: 32 и 16 нейронов для обработки данных
// Выходной слой (8): управляющие сигналы для 8 сервоприводов
const std::vector<int> topology = {24, 32, 16, 8};

using namespace webots; // Добавьте эту строку

int main(int argc, char **args)
{
}