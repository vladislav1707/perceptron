#include <C:/Program Files/Webots/include/controller/cpp/webots/Robot.hpp>
#include <C:/Program Files/Webots/include/controller/cpp/webots/Motor.hpp>
#include <C:/Program Files/Webots/include/controller/cpp/webots/Gyro.hpp>
#include <C:/Program Files/Webots/include/controller/cpp/webots/PositionSensor.hpp>
#include <C:/Program Files/Webots/include/controller/cpp/webots/TouchSensor.hpp>
#include "../perceptron.hpp"
#include "rewardManager.hpp"

using namespace webots;

double NetConfig::GRADIENT_CLIP = 1.0;
double NetConfig::BIAS = 1.0;
double NetConfig::learningRate = 0.01;
double NetConfig::LEAK_FACTOR = 0.01;

// Топология сети для GhostDog:
// Входной слой (21):
// - 4 значения с TouchSensor
// - 3 значения с Gyro (ориентация)
// - 12 значений с PositionSensor (положение суставов)
// - 1 значение расстояния до награды
// - 1 значение угла до награды
// Скрытые слои:
// - 32 нейрона для обработки сенсорных данных и пространственной информации
// - 24 нейрона для формирования управляющих сигналов
// Выходной слой (12): управляющие сигналы для 12 основных суставов
const std::vector<int> topology = {21, 32, 24, 12};

class RobotConfig
{
public:
    // Базовая конфигурация симуляции
    static const int TIME_STEP = 32; // Временной шаг симуляции в миллисекундах
                                     // 32мс = ~30 Гц частота обновления

    // Конфигурация сенсоров
    static const int SENSOR_SAMPLING_PERIOD = TIME_STEP; // Частота опроса сенсоров

    // Конфигурация моторов
    static const inline std::vector<std::string> motorNames = {"neck0",
                                                               "neck1",
                                                               "head",
                                                               "hip0",
                                                               "knee0",
                                                               "hip1",
                                                               "knee1",
                                                               "hip2",
                                                               "knee2",
                                                               "hip3",
                                                               "knee3",
                                                               "spine"};

    // Конфигурация сенсоров касания
    static const inline std::vector<std::string> TOUCH_SENSOR_NAMES = {
        "touch0", "touch1", "touch2", "touch3"};
};

// Вспомогательная функция для получения состояния робота
std::vector<double> getRobotState(const std::vector<TouchSensor *> &touchSensors,
                                  Gyro *gyro,
                                  const std::vector<Motor *> &motors,
                                  RewardManager &rewardManager)
{
    std::vector<double> state;

    // Получаем данные с сенсоров касания (бинарные значения)
    for (auto &sensor : touchSensors)
    {
        state.push_back(sensor->getValue());
    }

    // Получаем данные с гироскопа
    const double *gyroValues = gyro->getValues();
    for (int i = 0; i < 3; i++)
    {
        state.push_back(gyroValues[i]);
    }

    // Получаем положения моторов (в радианах)
    for (auto &motor : motors)
    {
        PositionSensor *sensor = motor->getPositionSensor();
        state.push_back(sensor->getValue());
    }

    // Информация о награде (в метрах и радианах)
    state.push_back(rewardManager.getDistanceToReward());
    state.push_back(rewardManager.getAngleToReward());

    return state;
}

int main(int argc, char **args)
{
    Robot robot;
    Supervisor supervisor;
    RewardManager rewardManager(&supervisor);
    Network net(topology, 4);

    // Инициализация всех моторов
    std::vector<Motor *> motors;
    for (const auto &motorName : RobotConfig::motorNames)
    {
        Motor *motor = robot.getMotor(motorName);
        motors.push_back(motor);

        // Включаем позиционный сенсор для каждого мотора
        PositionSensor *sensor = motor->getPositionSensor();
        sensor->enable(RobotConfig::TIME_STEP);

        // Устанавливаем мотор в режим качания
        motor->setPosition(INFINITY);
        motor->setVelocity(0.0);
    }

    // Инициализация гироскопа
    Gyro *gyro = robot.getGyro("gyro");
    gyro->enable(RobotConfig::TIME_STEP);

    // Инициализация сенсоров касания
    TouchSensor *touch0 = robot.getTouchSensor("touch0");
    TouchSensor *touch1 = robot.getTouchSensor("touch1");
    TouchSensor *touch2 = robot.getTouchSensor("touch2");
    TouchSensor *touch3 = robot.getTouchSensor("touch3");

    // Включение сенсоров касания
    touch0->enable(RobotConfig::TIME_STEP);
    touch1->enable(RobotConfig::TIME_STEP);
    touch2->enable(RobotConfig::TIME_STEP);
    touch3->enable(RobotConfig::TIME_STEP);

    // Создание вектора сенсоров касания
    std::vector<TouchSensor *> touchSensors = {
        touch0, touch1, touch2, touch3};

    // переменные для ε-жадной стратегии
    double epsilon = 0.1; // Вероятность выбора случайного действия
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Применяем выходные значения к моторам
    // Масштабируем выходы нейросети (-1 до 1) к скоростям мотора
    const double MAX_VELOCITY = 2.0; // максимальная скорость мотора в рад/с

    // первая награда
    rewardManager.spawnNewReward();

    while (robot.step(RobotConfig::TIME_STEP) != -1)
    {
        // Получаем текущее состояние робота
        std::vector<double> currentState = getRobotState(touchSensors, gyro, motors, rewardManager);

        // Получаем Q-значения для текущего состояния
        net.forward(currentState);
        std::vector<double> currentQ = net.getOutput();

        // Выбор действия
        int action;
        if (dis(gen) < epsilon)
        {
            // Случайное действие
            action = std::uniform_int_distribution<>(0, motors.size() - 1)(gen);
        }
        else
        {
            // Действие с максимальным Q-значением
            action = std::distance(currentQ.begin(), std::max_element(currentQ.begin(), currentQ.end()));
        }

        // Применяем выбранное действие к моторам
        double velocity = currentQ[action] * MAX_VELOCITY;
        motors[action]->setVelocity(velocity);

        // Получаем награду и проверяем столкновение
        bool done = rewardManager.checkCollision();
        double reward = done ? 1.0 : 0.0; // Пример награды

        // Получаем новое состояние
        std::vector<double> nextState = getRobotState(touchSensors, gyro, motors, rewardManager);

        // Обновляем сеть с использованием Q-обучения
        net.qLearn(currentState, action, reward, nextState, done);

        // Выводим текущий счет
        std::cout << rewardManager.getScore() << std::endl;
    }

    return 0;
}