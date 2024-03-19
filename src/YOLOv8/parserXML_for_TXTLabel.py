import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET
import os
import shutil

# ФУНКЦИЯ ВЫБОРА ПАПКИ С ИЗОБРАЖЕНИЯМИ
def select_image_folder():
    image_folder = filedialog.askdirectory()
    image_folder_input.delete(0, tk.END)
    image_folder_input.insert(0, image_folder)

# ФУНКИЯ ВЫБОРА XML-ФАЙЛА(ФАЙЛА АННОТАЦИИ)
def select_xml_file():
    xml_file = filedialog.askopenfilename(filetypes=[("XML Files", "*.xml")])
    xml_file_input.delete(0, tk.END)
    xml_file_input.insert(0, xml_file)

# ФУНКЦИЯ НЕПОСРЕДСТВЕННОГО ПРЕОБРАЗОВАНИЯ XML ФАЙЛА В TXT ФАЙЛ ТИПА YOLO
def convert_xml_to_yolo():
    try:
        image_folder = image_folder_input.get()
        xml_file = xml_file_input.get()
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Создаем новую папку для сохранения изображений и файлов разметок
        new_folder_name = 'yolo_annotations'
        new_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), new_folder_name)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        # Копируем выбранные изображения в новую папку
        for image in root.findall('image'):
            filename = image.get('name')
            shutil.copy2(os.path.join(image_folder, filename), new_folder_path)

        # Создаем файлы разметок для каждого изображения в новой папке
        for image in root.findall('image'):
            filename = image.get('name')
            im_width = int(image.get('width'))
            im_height = int(image.get('height'))
            yolo_annotations = []

            for obj in image.findall('box'):
                xmin = float(obj.get('xtl'))
                ymin = float(obj.get('ytl'))
                xmax = float(obj.get('xbr'))
                ymax = float(obj.get('ybr'))

                xcenter = (xmin + xmax) / 2 / im_width
                ycenter = (ymin + ymax) / 2 / im_height
                width = (xmax - xmin) / im_width
                height = (ymax - ymin) / im_height

                yolo_line = "0 {} {} {} {}".format(xcenter, ycenter, width, height)
                yolo_annotations.append(yolo_line)

            # Сохраняем разметки в файл с тем же именем, но расширением .txt
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_file_path = os.path.join(new_folder_path, txt_filename)
            with open(txt_file_path, 'w') as f:
                for line in yolo_annotations:
                    f.write(line + '\n')

        result_label.config(text="Преобразование завершено успешно", fg="green")
    except Exception as e:
        result_label.config(text="Ошибка: " + str(e), fg="red")

# ФУНКЦИЯ ЦЕНТРИРОВАНИЯ ОКНА TKINTER
def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

window = tk.Tk()
window.title("Преобразование XML в формат YOLO")
window.geometry("600x200")
center_window(window)


image_folder_label = tk.Label(window, text="Папка с изображениями:")
image_folder_label.pack()

image_folder_input = tk.Entry(window, width=50)
image_folder_input.pack()

image_folder_button = tk.Button(window, text="Выбрать", command=select_image_folder)
image_folder_button.pack()

xml_file_label = tk.Label(window, text="XML-файл:")
xml_file_label.pack()

xml_file_input = tk.Entry(window, width=50)
xml_file_input.pack()

xml_file_button = tk.Button(window, text="Выбрать", command=select_xml_file)
xml_file_button.pack()

convert_button = tk.Button(window, text="Преобразовать в YOLO", command=convert_xml_to_yolo)
convert_button.pack()

result_label = tk.Label(window, text="", fg="black")
result_label.pack()

window.mainloop()