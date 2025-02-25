#ifndef REWARD_MANAGER_HPP
#define REWARD_MANAGER_HPP

#include <C:/Program Files/Webots/include/controller/cpp/webots/Supervisor.hpp>
#include <C:/Program Files/Webots/include/controller/cpp/webots/Node.hpp>
#include <C:/Program Files/Webots/include/controller/cpp/webots/Field.hpp>
#include <random>
#include <cmath>
#include <string>

class RewardManager
{
private:
    webots::Supervisor *supervisor;
    webots::Node *robotNode;
    webots::Node *rewardNode;
    double rewardSize;  // Размер награды
    double spawnRadius; // Максимальное расстояние от робота для спавна награды
    int score;

    // Генератор случайных чисел
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    std::string getRewardProtoString()
    {
        return "DEF REWARD Solid {\n"
               "  translation 0 0 0\n"
               "  children [\n"
               "    Shape {\n"
               "      appearance PBRAppearance {\n"
               "        baseColor 1 0.8 0\n" // Золотой цвет (RGB)
               "        metalness 0.8\n"     // Металлический эффект
               "        roughness 0.3\n"     // Гладкость поверхности
               "      }\n"
               "      geometry Box {\n"
               "        size " +
               std::to_string(rewardSize) + " " +
               std::to_string(rewardSize) + " " +
               std::to_string(rewardSize) + "\n"
                                            "      }\n"
                                            "    }\n"
                                            "  ]\n"
                                            "  boundingObject Box {\n"
                                            "    size " +
               std::to_string(rewardSize) + " " +
               std::to_string(rewardSize) + " " +
               std::to_string(rewardSize) + "\n"
                                            "  }\n"
                                            "  physics Physics {}\n"
                                            "}\n";
    }

public:
    RewardManager(webots::Supervisor *sup) : supervisor(sup),
                                             robotNode(nullptr),
                                             rewardNode(nullptr),
                                             rewardSize(0.1),
                                             spawnRadius(2.0),
                                             score(0),
                                             gen(rd()),
                                             dis(-spawnRadius, spawnRadius)
    {
        if (!supervisor)
        {
            std::cerr << "[RewardManager] Error: supervisor is null." << std::endl;
            return;
        }
        robotNode = supervisor->getSelf();
        if (!robotNode)
        {
            std::cerr << "[RewardManager] Warning: getSelf() returned nullptr."
                      << " The controller might not be attached to a robot node." << std::endl;
        }
    }

    // Создать новую награду в случайной позиции
    void spawnNewReward()
    {
        // 1. Правильно удаляем старую награду
        if (rewardNode)
        {
            rewardNode->remove();
            rewardNode = nullptr;
        }

        // 2. Получаем позицию робота и генерируем новые координаты
        const double *robotPos = robotNode->getPosition();
        double x = robotPos[0] + dis(gen);
        double z = robotPos[2] + dis(gen);

        // 3. Создаем новую награду через children field корневого узла
        webots::Field *rootChildren = supervisor->getRoot()->getField("children");
        rootChildren->importMFNodeFromString(-1, getRewardProtoString());

        // 4. Получаем созданный узел награды
        rewardNode = supervisor->getFromDef("REWARD");

        // 5. Устанавливаем позицию
        if (rewardNode)
        {
            double newPosition[3] = {x, rewardSize / 2, z};
            rewardNode->getField("translation")->setSFVec3f(newPosition);
        }
    }

    // Проверить столкновение с наградой
    bool checkCollision()
    {
        if (!rewardNode)
            return false;

        const double *robotPos = robotNode->getPosition();
        const double *rewardPos = rewardNode->getPosition();

        double dx = robotPos[0] - rewardPos[0];         // Разница по X
        double dz = robotPos[2] - rewardPos[2];         // Разница по Z
        double distance = std::sqrt(dx * dx + dz * dz); // Теорема Пифагора

        if (distance < (rewardSize + 0.2)) // 0.2 - радиус робота
        {
            score++;          // Увеличиваем счет
            spawnNewReward(); // Создаем новую награду
            return true;      // Сообщаем о столкновении
        }
        return false;
    }

    // Получить направление к награде (в радианах)
    double getAngleToReward()
    {
        if (!rewardNode)
            return 0.0;

        const double *robotPos = robotNode->getPosition();
        const double *rewardPos = rewardNode->getPosition();

        double dx = rewardPos[0] - robotPos[0];
        double dz = rewardPos[2] - robotPos[2];

        return std::atan2(dz, dx); // возвращает угол в радианах [-π, π]
    }

    // Получить расстояние до награды
    double getDistanceToReward()
    {
        if (!rewardNode)
            return -1.0;

        const double *robotPos = robotNode->getPosition();
        const double *rewardPos = rewardNode->getPosition();

        double dx = rewardPos[0] - robotPos[0]; // Разница по X
        double dz = rewardPos[2] - robotPos[2]; // Разница по Z

        return std::sqrt(dx * dx + dz * dz); // Теорема Пифагора
    }

    // Получить текущий счет
    int getScore() const
    {
        return score;
    }
};

#endif // REWARD_MANAGER_HPP