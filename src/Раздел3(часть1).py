# Задание1.
# Давайте попрактикуемся с SineNet:
#
# 1) Добавим еще один fc-слой
# 2) Заменим активацию между слоями на гиперболический тангенс

import torch  # Импорт библиотеки PyTorch
import torch.nn as nn  # Импорт модуля для создания нейронных сетей

# Определение класса SineNet, который наследуется от nn.Module
class SineNet(nn.Module):
    # Инициализация нейронной сети
    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()  # Вызов инициализации родительского класса
        # Создание первого полносвязного слоя с одним входом и n_hidden_neurons выходами
        self.fc1 = nn.Linear(1, n_hidden_neurons)
        # Создание гиперболического тангенса в качестве функции активации
        self.act1 = nn.Tanh()
        # Создание второго полносвязного слоя с n_hidden_neurons входами и выходами
        self.fc2 = nn.Linear(n_hidden_neurons, n_hidden_neurons)
        # Создание еще одной функции активации - гиперболического тангенса
        self.act2 = nn.Tanh()
        # Создание третьего полносвязного слоя с n_hidden_neurons входами и одним выходом
        self.fc3 = nn.Linear(n_hidden_neurons, 1)

    # Определение процесса прямого прохода (forward pass)
    def forward(self, x):
        x = self.fc1(x)  # Применение первого полносвязного слоя
        x = self.act1(x)  # Применение функции активации
        x = self.fc2(x)  # Применение второго полносвязного слоя
        x = self.act2(x)  # Применение функции активации
        x = self.fc3(x)  # Применение третьего полносвязного слоя
        return x  # Возврат выходного значения

# Создание экземпляра SineNet с 20 скрытыми нейронами
sine_net = SineNet(20)
# Вывод структуры нейронной сети
print(sine_net)

# Задание 2
# Обучим нейронную сеть для задачи регрессии:
# Возьмем более сложную функцию в качестве таргета: y=2^x*sin(2^-x).
#
# Кроме того, мы хотим получить хорошую метрику MAE на валидации: MAE = 1/l сумма от i до l |y_predi - y_targei|,
# тогда как знакомая нам MSE выглядит как MAE = 1/l сумма от i до l(y_predi - y_targei)^2
#
#Данный пример показывает MAE на валидации ~0.021. Получите метрику не хуже 0.03

#Что можно варьировать:
#1) Архитектуру сети
#2) loss-функцию
#3) lr оптимизатора
#4) Количество эпох в обучении

import matplotlib.pyplot as plt  # Импорт библиотеки matplotlib для визуализации

# Определение целевой функции
def target_function(x):
    return 2**x * torch.sin(2**-x)

# Определение класса нейронной сети для регрессии
class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(RegressionNet, self).__init__()
        # Первый полносвязный слой с одним входом и n_hidden_neurons скрытыми нейронами
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        # Функция активации - гиперболический тангенс
        self.act1 = torch.nn.Tanh()
        # Второй полносвязный слой с n_hidden_neurons входами и одним выходом
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    # Определение процесса прямого прохода (forward pass)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

# Создание экземпляра нейронной сети с 50 скрытыми нейронами
net = RegressionNet(50)

# ------Начало подготовки датасета--------:
# Генерация тренировочных данных
x_train = torch.linspace(-10, 5, 1000)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

# Генерация валидационных данных
x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)
# ------Конец подготовки датасета--------:

# Инициализация оптимизатора Adam с шагом обучения 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# Использование среднеквадратичной ошибки в качестве функции потерь
loss = torch.nn.MSELoss()

# Обучение нейронной сети
for epoch_index in range(5000):
    optimizer.zero_grad()  # Обнуление градиентов
    y_pred = net.forward(x_train)  # Прямой проход
    loss_value = loss(y_pred, y_train)  # Вычисление функции потерь
    loss_value.backward()  # Обратное распространение ошибки
    optimizer.step()  # Обновление весов

# Функция для вычисления метрики MAE
def metric(pred, target):
    return (pred - target).abs().mean()

# Визуализация
x_range = torch.linspace(-10, 5, 100)  # Генерация точек для визуализации
y_range = target_function(x_range)  # Вычисление значений целевой функции
y_predicted = net.forward(x_range.unsqueeze(1))  # Предсказания нейронной сети

# Построение графика целевой функции и ее предсказаний
plt.plot(x_range.detach().numpy(), y_range.detach().numpy(), label='Target function')
plt.plot(x_range.detach().numpy(), y_predicted.detach().numpy(), label='Predicted function')
plt.legend()  # Добавление легенды
plt.show()  # Отображение графика

# Вывод метрики MAE на валидации
print(metric(net.forward(x_validation), y_validation).item())
