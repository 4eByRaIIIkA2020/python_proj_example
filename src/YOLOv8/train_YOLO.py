from ultralytics import YOLO

# Путь к YAML файлу с описанием датасета
data_yaml = 'dataset.yaml'

# Параметры для первой итерации обучения
epochs = 45  # Количество эпох обучения
imgsz = 640  # Размер изображения для обучения
batch = 20    # Размер батча для обучения

try:
    # Загрузка предварительно обученной модели
    model = YOLO('yolov8n.pt')
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch)

    # Сохранение весов обученной модели
    model.save('yolov8n_iteration_3.pt')
    print("Обучение успешно завершено.")
except Exception as e:
    print(f"Произошла ошибка во время обучения: {e}")