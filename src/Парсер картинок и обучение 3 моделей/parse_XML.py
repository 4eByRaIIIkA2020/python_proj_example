import xml.etree.ElementTree as ET
from PIL import Image
import os

def parse_and_crop(xml_file, image_dir, output_dir):
    # Создаем директорию для выходных файлов, если её ещё нет
    os.makedirs(output_dir, exist_ok=True)

    # Разбираем XML-файл
    tree = ET.parse(xml_file)
    root = tree.getroot()

    processed_labels = 0 # счетчик обработанных меток
    schet = 0 # счетчик количества найденных меток

    # Проходим по каждому изображению в XML
    for image_elem in root.findall('image'):
        # Получаем имя изображения
        image_name = image_elem.get('name')
        # Формируем путь к изображению
        image_path = os.path.join(image_dir, image_name)

        # Проверяем, существует ли файл изображения
        if not os.path.isfile(image_path):
            print(f"Предупреждение: Файл изображения {image_name} не найден. Пропускаем...")
            continue

        # Открываем изображение
        try:
            img = Image.open(image_path)
        except IOError as e:
            print(f"Ошибка открытия изображения {image_name}: {e}")
            continue

        # Проходим по каждому ограничивающему прямоугольнику в изображении
        for box_elem in image_elem.findall('box'):
            # Получаем метку (название) ограничивающего прямоугольника
            label = box_elem.get('label')
            schet += 1

            # Получаем координаты ограничивающего прямоугольника
            xtl = float(box_elem.get('xtl'))
            ytl = float(box_elem.get('ytl'))
            xbr = float(box_elem.get('xbr'))
            ybr = float(box_elem.get('ybr'))

            # Обрезаем изображение по ограничивающему прямоугольнику
            try:
                cropped_img = img.crop((xtl, ytl, xbr, ybr))
            except Exception as e:
                print(f"Ошибка обрезки изображения {image_name} с меткой {label}: {e}")
                continue

            # Формируем имя для обрезанного изображения
            output_filename = f'{label}{schet}_{image_name}'
            # Формируем полный путь для сохранения обрезанного изображения
            output_path = os.path.join(output_dir, output_filename)

            # Сохраняем обрезанное изображение
            try:
                cropped_img.save(output_path)
                # Увеличиваем счетчик обработанных меток
                processed_labels += 1
            except IOError as e:
                print(f"Ошибка сохранения обрезанного изображения {output_filename}: {e}")

    # Выводим общее количество обработанных меток
    print(f"Обработано меток: {processed_labels}")


# Парсер самолётов
#xml_file = 'annotations/annotations_aircrafts.xml'
#image_dir = 'images/aircrafts'
#output_dir = 'output/aircrafts/'

# Парсер кораблей
xml_file = 'annotations/annotations_boats.xml'
image_dir = 'images/boats'
output_dir = 'output/boats/'
parse_and_crop(xml_file, image_dir, output_dir)