# Документация нейронной сети

## Описание
Данная нейронная сеть представляет собой многослойный перцептрон с возможностью настройки топологии и параметров обучения. Сеть использует leaky ReLU в качестве функции активации и поддерживает сохранение/загрузку обученной модели. Помимо решения MNIST тут есть файл для управления роботом ghostDog в webots

## Настраиваемые параметры

### Топология сети
Топологию сети можно настроить, изменяя вектор `topology`. Каждое число в векторе представляет количество нейронов в соответствующем слое:
std::vector<int> topology = {входной_слой, скрытый_слой1, скрытый_слой2, ..., выходной_слой};

### Параметры обучения
learningRate: Скорость обучения сети. Влияет на величину изменения весов при обучении. Рекомендуется начать с небольших значений (например, 0.001 - 0.1)
BIAS: Смещение нейронов. По умолчанию установлено значение 1.0
Основные методы
Создание сети
Network network(topology);

## Прямое распространение
network.forward(input_vector);

Принимает вектор входных данных размером, соответствующим первому слою в топологии.

## Обратное распространение
network.backprop(target_vector);

Принимает вектор целевых значений размером, соответствующим последнему слою в топологии.

Получение результата
std::vector<double> output = network.getOutput();

Возвращает вектор выходных значений сети.

## Сохранение и загрузка модели
### Сохранение
network.saveToFile("model.bin");

### Загрузка
Network* loadedNetwork = Network::loadFromFile("model.bin");

# Рекомендации по настройке
## Топология
Входной слой должен соответствовать размерности входных данных
Можно экспериментировать с количеством скрытых слоев и нейронов в них
Выходной слой должен соответствовать количеству классов или размерности выходных данных
Обучение
Начните с небольшой скорости обучения
Следите за ошибкой с помощью метода calculateError()
При переобучении уменьшите скорость обучения или измените топологию

данная документация написана искуственным интеллектом и слабо проверялась(а так же написана немножко давно), так же можно легко менять функцию активации

осторожно, могут быть изменения которых нет в документации например добавление RL

***в проекте для обучения использовался датасет MNIST***