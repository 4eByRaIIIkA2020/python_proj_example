import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
from torch.utils.tensorboard import SummaryWriter

# Определяем пути к тренировочному и тестовому набору данных
train_path = "aircrafts_vs_boats/train"
test_path = "aircrafts_vs_boats/test"

# Определяем преобразования изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загружаем наборы данных с определенными преобразованиями
train_data = dataset.ImageFolder(train_path, transform)
test_data = dataset.ImageFolder(test_path, transform)

# Создаем загрузчики данных для пакетной обработки
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)


# Функция для вычисления точности
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# Функция для обучения модели за одну эпоху
def train(model, dataloader, optimizer, loss_function, device):
    epoch_acc = 0
    epoch_loss = 0
    model.train()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predicts = model(images)
        loss = loss_function(predicts, labels)
        acc = calculate_accuracy(predicts, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


# Функция для оценки модели за одну эпоху
def evaluate(model, dataloader, loss_function, device):
    epoch_acc = 0
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            predicts = model(images)
            loss = loss_function(predicts, labels)
            acc = calculate_accuracy(predicts, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


# Определяем устройство (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры обучения
epochs = 5  # Количество эпох
loss_function = nn.CrossEntropyLoss()  # Функция потерь

# Модели для обучения: ResNet-18, ResNet-34, и ResNet-50
models_to_train = {
    "resnet18": models.resnet18(pretrained=True),
    "resnet34": models.resnet34(pretrained=True),
    "resnet50": models.resnet50(pretrained=True)
}

# Инициализируем SummaryWriter для TensorBoard
writer = SummaryWriter()


# Функция для вычисления времени эпохи
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Цикл обучения для каждой модели
for name, model in models_to_train.items():
    print(f"Обучение модели {name}...")
    model.to(device)

    # Инициализируем оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    best_acc = 0

    for epoch in range(epochs):
        start_time = time.monotonic()
        train_loss, train_acc = train(model, train_loader, optimizer, loss_function, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_function, device)
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

        if test_loss < best_loss:
            best_loss = test_loss
            best_acc = test_acc
            torch.save(model.state_dict(), f"{name}_best_model.pt")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc * 100, epoch)
    writer.add_scalar("Accuracy/test", test_acc * 100, epoch)

    print(f"Модель {name} обучена. Лучшая точность на тестовом наборе данных: {best_acc * 100:.2f}%")
