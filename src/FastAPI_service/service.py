import numpy as np
import io
import json
import logging
import uvicorn
from fastapi import FastAPI, File, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image
from classifier import Classifier
from detector import Detector
from datacontract.service_config import ServiceConfig
from datacontract.service_output import *

# Настройка логгера
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Создание экземпляра FastAPI приложения
app = FastAPI()

# Путь к файлу конфигурации сервиса
service_config_path = "configs\\service_config.json"

# Открытие и чтение файла конфигурации сервиса
with open(service_config_path, "r") as service_config:
    service_config_json = json.load(service_config)

# Создание адаптера для валидации данных конфигурации сервиса
service_config_adapter = pydantic.TypeAdapter(ServiceConfig)

# Валидация и получение данных конфигурации сервиса
service_config_python = service_config_adapter.validate_python(service_config_json)

# Словарь для сопоставления индексов классов и их названий
class_names = {0: "aircraft", 1: "ship"}

# Создание экземпляра классификатора с указанным путем к модели и словарем имен классов
classifier = Classifier(service_config_python.path_to_classifier, class_names)
logger.info(f"Загружен классификатор с путем: {service_config_python.path_to_classifier}")

# Создание экземпляра детектора с указанным путем к модели
detector = Detector(service_config_python.path_to_detector)
logger.info(f"Загружен детектор с путем: {service_config_python.path_to_detector}")


# Определение эндпоинта для проверки состояния сервиса
@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Проверка состояния сервиса",
    response_description="Возвращает HTTP статус 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def health_check() -> str:
    return '{"Status" : "OK"}'


# Определение эндпоинта для выполнения инференса на изображении
@app.post("/file/")
async def inference(image: UploadFile = File(...)) -> JSONResponse:
    # Чтение содержимого изображения
    image_content = await image.read()

    # Открытие изображения с помощью PIL
    pil_image = Image.open(io.BytesIO(image_content))

    # Преобразование изображения в формат RGB, если оно не в формате RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Преобразование изображения в массив numpy
    cv_image = np.array(pil_image)
    logger.info(f"Принята картинка размерности: {cv_image.shape}")

    # Инициализация словаря для хранения обнаруженных объектов
    output_dict = {"objects": []}

    # Выполнение детекции на изображении
    detector_outputs = detector.detect(cv_image)

    # Обработка результатов детекции
    for res in detector_outputs:
        boxes = res.boxes
        for box in boxes:
            # Получение координат ограничивающего прямоугольника
            cords = box.xyxy
            xtl, ytl, xbr, ybr = int(cords[0][0]), int(cords[0][1]), int(cords[0][2]), int(cords[0][3])

            # Вырезание области изображения, соответствующей ограничивающему прямоугольнику
            crop_object = cv_image[ytl:ybr, xtl:xbr]

            # Классификация области изображения
            class_name = classifier.classify(Image.fromarray(crop_object))

            # Добавление обнаруженного объекта в словарь
            output_dict["objects"].append(DetectedObject(xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr, class_name=class_name))
            logger.info(f"Обнаружен объект: {class_name} с координатами: ({xtl}, {ytl}), ({xbr}, {ybr})")

    # Создание объекта выхода сервиса
    service_output = ServiceOutput(objects=output_dict["objects"])

    # Преобразование объекта выхода сервиса в JSON
    service_output_json = service_output.model_dump(mode="json")

    # Запись JSON в файл
    with open("output_json.json", "w") as output_file:
        json.dump(service_output_json, output_file, indent=4)

    # Возврат JSON-ответа
    return JSONResponse(content=jsonable_encoder(service_output_json))

# Замена команды uvicorn src.service:app --host localhost --port 8000 --log-config=log_config.yaml
# Запускает FastAPI после выполнения кода
uvicorn.run(app, host="0.0.0.0", port=8000)