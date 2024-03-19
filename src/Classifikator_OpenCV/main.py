import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms
import os
import torch.nn as nn

classifier = None
device = None
image_folder = None
image_files = []
image_index = 0

# ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ МОДЕЛИ КЛАССИФИКАТОРА
def load_classifier_model(pth_file, device, model_type):
    # Инициализация выбранной модели
    if model_type == "resnet18":
        model = models.resnet18(pretrained=False)
    elif model_type == "resnet34":
        model = models.resnet34(pretrained=False)
    else:  # По умолчанию используется ResNet50
        model = models.resnet50(pretrained=False)

    # Изменение последнего слоя для соответствия количеству классов
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(500, 2)
    )

    # Загрузка состояния словаря
    state_dict = torch.load(pth_file, map_location=device).state_dict()

    # Загрузка состояния словаря в модель
    model.load_state_dict(state_dict)

    # Перемещение модели на выбранное устройство
    model.to(device)

    # Возврат загруженной модели
    return model

# ФУНКЦИЯ ДЛЯ КЛАССИФИКАЦИИ ИЗОБРАЖЕНИЯ
def inference_classifier(classifier, image_path):
    # Загрузка и предварительная обработка изображения
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = image.convert('RGB')
    transform = transforms.ToTensor()
    tensor_image = transform(image).unsqueeze(0)

    # Классификация изображения
    with torch.no_grad():
        output = classifier(tensor_image)
    _, predicted_idx = torch.max(output, 1)
    class_index = predicted_idx.item()
    class_labels = ["самолет", "корабль"]  # Замените на реальные метки классов
    predicted_class = class_labels[class_index]

    # Преобразование PIL Image в Tkinter PhotoImage
    photo_image = ImageTk.PhotoImage(image)
    return predicted_class, photo_image

# ФУНКЦИЯ ДЛЯ ОТОБРАЖЕНИЯ ИЗОБРАЖЕНИЯ И КЛАССА
def display_image_and_class(photo_image, predicted_class):
    # Обновление метки изображения
    image_label.config(image=photo_image)
    image_label.image = photo_image

    # Обновление метки класса
    class_label.config(text=f"Класс: {predicted_class}")

# ФУНКЦИЯ ДЛЯ НАВИГАЦИИ ПО ИЗОБРАЖЕНИЯМ
def navigate_images(direction):
    global image_index, image_files, image_folder
    if image_folder is not None and image_files:
        if direction == -1 and image_index > 0:
            image_index -= 1
        elif direction == 1 and image_index < len(image_files) - 1:
            image_index += 1
        image_path = os.path.join(image_folder, image_files[image_index])
        predicted_class, photo_image = inference_classifier(classifier, image_path)
        display_image_and_class(photo_image, predicted_class)

# ФУНКЦИЯ ДЛЯ ВЫБОРА PTH-ФАЙЛА
def select_pth_file():
    global classifier, device, pth_file_path
    pth_file_path = filedialog.askopenfilename(filetypes=[("PTH files", "*.pth")])
    pth_file_entry.delete(0, tk.END)
    pth_file_entry.insert(0, pth_file_path)
    if pth_file_path:
        device = torch.device("cuda" if torch.cuda.is_available() and cuda_var.get() == "CUDA" else "cpu")
        classifier = load_classifier_model(pth_file_path, device, model_type_var.get())
        classifier.eval()

# ФУНКЦИЯ ДЛЯ ВЫБОРА ПАПКИ С ИЗОБРАЖЕНИЯМИ
def select_image_folder():
    global image_folder, image_files, image_index, image_folder_path
    image_folder_path = filedialog.askdirectory()
    image_folder_entry.delete(0, tk.END)
    image_folder_entry.insert(0, image_folder_path)
    if image_folder_path:
        image_folder = image_folder_path
        image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        image_index = 0
        if image_files:
            navigate_images(0)

# ФУНКЦИЯ ДЛЯ ЦЕНТРИРОВАНИЯ ОКНА TKINTER НА ЭКРАНЕ
def center_window(root, width=1200, height=600):
    # Получение ширины и высоты экрана
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Вычисление координат x и y для центрирования окна
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.geometry(f'{width}x{height}+{int(x)}+{int(y)}')

    # Делаем окно полноэкранным
    root.attributes('-fullscreen', True)

root = tk.Tk()
root.title("Классификатор изображений")

# Центрирование окна на экране и делаем его полноэкранным
center_window(root)

# Создание рамки для элементов управления
frame = tk.Frame(root)
frame.pack(fill=tk.X)

tk.Label(frame, text="Выберите тип модели:").pack(side=tk.LEFT)
model_type_var = tk.StringVar()
model_type_var.set("resnet50")
model_type_menu = tk.OptionMenu(frame, model_type_var, "resnet18", "resnet34", "resnet50")
model_type_menu.pack(side=tk.LEFT)

tk.Label(frame, text="Выберите PTH файл:").pack(side=tk.LEFT)
pth_file_entry = tk.Entry(frame, width=50)
pth_file_entry.pack(side=tk.LEFT)

tk.Button(frame, text="Обзор", command=select_pth_file).pack(side=tk.LEFT)

tk.Label(frame, text="Выберите устройство:").pack(side=tk.LEFT)
cuda_var = tk.StringVar()
cuda_var.set("CUDA" if torch.cuda.is_available() else "CPU")
cuda_menu = tk.OptionMenu(frame, cuda_var, "CUDA", "CPU")
cuda_menu.pack(side=tk.LEFT)

image_folder_label = tk.Label(frame, text="Путь к папке с изображениями:")
image_folder_label.pack(side=tk.LEFT, padx=(10, 5))
image_folder_entry = tk.Entry(frame, width=50)
image_folder_entry.pack(side=tk.LEFT)

image_folder_button = tk.Button(frame, text="Обзор", command=select_image_folder)
image_folder_button.pack(side=tk.LEFT, padx=(5, 10))

image_frame = tk.Frame(root)
image_frame.pack(fill=tk.BOTH, expand=True)

image_label = tk.Label(image_frame)
image_label.pack(fill=tk.BOTH, expand=True)

class_label = tk.Label(image_frame, text="Класс:", font=("Helvetica", 24))
class_label.pack(side=tk.BOTTOM, pady=20)

button_frame = tk.Frame(image_frame)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

prev_button = tk.Button(button_frame, text="Предыдущее", command=lambda: navigate_images(-1), font=("Helvetica", 16))
prev_button.pack(side=tk.LEFT, padx=(10, 5))

next_button = tk.Button(button_frame, text="Следующее", command=lambda: navigate_images(1), font=("Helvetica", 16))
next_button.pack(side=tk.RIGHT, padx=(5, 10))

exit_button = tk.Button(root, text="Выйти из программы", command=root.quit, font=("Helvetica", 16))
exit_button.pack(side=tk.BOTTOM, pady=20)

root.mainloop()