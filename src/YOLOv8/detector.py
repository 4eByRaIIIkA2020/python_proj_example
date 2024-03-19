import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO

# Загрузите вашу обученную модель YOLOv8
model = YOLO('yolov8n_iteration_2.pt')

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = cv2.imread(file_path)
        results = model(image)
        # Отрисовка рамки только вокруг обнаруженных объектов на изображении с помощью OpenCV
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imshow('Детектор YOLOv8', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

root = tk.Tk()
root.title('Обнаружение объектов YOLOv8')

select_button = tk.Button(root, text='Выбрать изображение', command=select_image)
select_button.pack()

root.mainloop()