import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from tkinter import Tk, Label
from PIL import Image, ImageTk
from tkinter import ttk

# 加载模型
model_list = glob.glob('/mnt/e/CatFeeder/AI/Source/*.keras')

for i, path in enumerate(model_list):
    print(f'{i}: {os.path.basename(path)}')

model_path = model_list[int(input('Choose a model: '))]

model = tf.keras.models.load_model(model_path)

# 指定图像目录
directory = '/mnt/e/CatFeeder/AI/TrainingData/Cat/MyCats/'  # 修改为您的图像目录路径
image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]
current_index = 0  # 初始化当前显示的图片索引

def update_image(index):
    global current_index
    current_index = index % len(image_files)  # 确保索引在有效范围内
    image_path = os.path.join(directory, image_files[current_index])
    predict_and_display_image(image_path)

def predict_and_display_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))
    frame = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)

    pred = model.predict(frame)
    pred = pred * 224

    for i in range(0, len(pred[0]), 2):
        x, y = int(pred[0][i]), int(pred[0][i+1])
        cv2.circle(image_resized, (x, y), 2, (0, 255, 0), -1)

    img_pil = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(img_pil)
    label.config(image=img_tk)
    label.image = img_tk

def on_key(event):
    if event.keysym == 'Left':
        update_image(current_index - 1)  # 上一张图片
    elif event.keysym == 'Right':
        update_image(current_index + 1)  # 下一张图片

# 创建Tkinter窗口
root = Tk()
root.title('Image Viewer')
root.geometry('300x300')  # 窗口大小

label = Label(root)
label.pack(expand=True, fill='both')

root.bind('<Left>', on_key)  # 绑定左键事件
root.bind('<Right>', on_key)  # 绑定右键事件

update_image(current_index)  # 显示第一张图片

root.mainloop()
