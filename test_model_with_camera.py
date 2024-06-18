import os
import cv2
import glob
import keras
import numpy as np
import tensorflow as tf

model_list = glob.glob('/mnt/e/CatFeeder/AI/Source/*.keras')

for i, path in enumerate(model_list):
    print(f'{i}: {os.path.basename(path)}')

model_path = model_list[int(input('Choose a model: '))]

model = keras.models.load_model(model_path)

ret = True

W=640
H=360
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, 30)

while ret:
    ret, frame = cap.read()

    if not ret:
        print('Failed to capture image')
        break

    # 保存原始帧的副本用于显示
    display_frame = cv2.resize(frame, (224, 224))
    
    # 对帧进行预处理并预测
    frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)  # 这里使用np.expand_dims更清晰

    pred = model.predict(frame)
    pred = pred * 224  # 将预测的坐标缩放回原始尺寸

    # 绘制预测的关键点
    for i in range(0, len(pred[0]), 2):
        x, y = int(pred[0][i]), int(pred[0][i+1])
        cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)

    # 确保显示帧的数据类型正确
    display_frame = np.clip(display_frame, 0, 255).astype('uint8')

    cv2.imshow('frame', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
