#include <webots/Robot.hpp>
#include <webots/Motor.hpp>
#include <webots/DistanceSensor.hpp>
#include <webots/Gyro.hpp>
#include <webots/PositionSensor.hpp>
#include "../perceptron.hpp"
#include "rewardManager.hpp"

using namespace webots;

double NetConfig::GRADIENT_CLIP = 1.0;
double NetConfig::BIAS = 1.0;
double NetConfig::learningRate = 0.01;
double NetConfig::LEAK_FACTOR = 0.01;

// Топология сети для GhostDog:
// Входной слой (15):
// - 4 значения с TouchSensor
// - 3 значения с Gyro (ориентация)
// - 8 значений с PositionSensor (положение суставов)
// Скрытые слои: 16 нейронов для обработки данных
// Выходной слой (8): управляющие сигналы для 8 основных суставов
const std::vector<int> topology = {15, 16, 8};

class robotController
{
    std::vector<std::string> motorNames = {"neck0",
                                           "neck1",
                                           "head",
                                           "hip0",
                                           "knee0",
                                           "hip2",
                                           "knee2",
                                           "hip1",
                                           "knee1",
                                           "hip3",
                                           "knee3",
                                           "spine"};
};

int main(int argc, char **args)
{
    Robot robot;
    Network net(topology, 4);
}