# Фиксирование seed не гарантирует, что код будет одинаково выполняться на разных компьютерах.
# Но на одной и той же машине вы будете получать одинаковые результаты, перезапуская один и тот же скрипт.
# Например, функция random.randint(start, end) отдает случайное целое число в диапазоне от start, end (включительно).
# Запуская скрипт, состоящий из вызова этой функции, вы будете получать разные ответы.
# В этом задании вам нужно подобрать seed, чтобы функция random.randint(0, 10) выдала число 5

import random
random.seed(7)
print(random.randint(0, 10))



# Давайте попрактикуемся с WineNet. Измените архитектуру так, чтобы на вход принимались все 13 признаков

import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

# Загрузка датасета вин
wine = load_wine()
features = 13  # Количество признаков

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    wine.data[:, :features],
    wine.target,
    test_size=0.3,
    shuffle=True)

# Преобразование данных в тензоры PyTorch
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

class WineNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden_neurons):
        super(WineNet, self).__init__()
        # Инициализация первого полносвязного слоя с функцией активации Sigmoid
        self.fc1 = torch.nn.Linear(n_input, n_hidden_neurons)
        self.activ1 = torch.nn.Sigmoid()
        # Инициализация второго полносвязного слоя
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.activ2 = torch.nn.Sigmoid()
        # Инициализация третьего полносвязного слоя
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 3)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)  # Применение первого полносвязного слоя
        x = self.activ1(x)  # Применение функции активации Sigmoid к выходным значениям первого слоя
        x = self.fc2(x)  # Применение второго полносвязного слоя
        x = self.activ2(x)
        x = self.fc3(x)
        return x  # Возврат выходных значений модели

    def inference(self, x):
        # Инференс модели
        x = self.forward(x)  # Прямой проход модели
        x = self.sm(x)  # Применение функции активации Softmax к выходным значениям модели
        return x  # Возврат выходных значений модели после применения Softmax

# Инициализация параметров модели
n_input = 13  # Количество входных нейронов
n_hidden = 2  # Количество скрытых нейронов
wine_net = WineNet(n_input, n_hidden)

# Инициализация функции потерь и оптимизатора
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(wine_net.parameters(), lr=1.0e-3)
batch_size = 10  # Размер батча

# Инициализация списков для хранения истории обучения
train_loss_history = []
test_loss_history = []

# Эпохи обучения
for epoch in range(2000):
    # Создание случайной перестановки индексов обучающей выборки
    order = np.random.permutation(len(X_train))
    # Цикл по батчам обучающей выборки
    for start_index in range(0, len(X_train), batch_size):
        # Обнуление градиентов оптимизатора
        optimizer.zero_grad()

        # Выбор индексов для текущего батча
        batch_indexes = order[start_index:start_index+batch_size]

        # Выбор данных для текущего батча
        x_batch = X_train[batch_indexes]
        y_batch = y_train[batch_indexes]

        # Предсказание модели для текущего батча
        preds = wine_net.forward(x_batch)

        # Вычисление значения функции потерь
        loss_value = loss(preds, y_batch)
        # Вычисление градиентов функции потерь
        loss_value.backward()

        # Обновление параметров модели
        optimizer.step()

    # Вычисление потерь на обучающей и тестовой выборках
    train_preds = wine_net.forward(X_train)
    train_loss = loss(train_preds, y_train)
    train_loss_history.append(train_loss.item())

    test_preds = wine_net.forward(X_test)
    test_loss = loss(test_preds, y_test)
    test_loss_history.append(test_loss.item())

# Визуализация истории обучения
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Тренировочные потери')
plt.plot(test_loss_history, label='Tестовые потери')
plt.title('Тренировочные и тестовые потери. История')
plt.xlabel('Эпоха')
plt.ylabel('Потеря')
plt.legend()
plt.show()

# Вывод количества входных нейронов и точности предсказаний
test_preds = wine_net.forward(X_test)
test_preds = test_preds.argmax(dim=1)
print(wine_net.fc1.in_features, np.asarray((test_preds == y_test).float().mean()) > 0.8)