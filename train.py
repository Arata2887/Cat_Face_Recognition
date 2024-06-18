import os
import cv2
import tqdm
import glob
import keras
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from feature_extractor import FeatureExtractionModel, MyCNNFeatureExtractionModel
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

EPOCHS = 1000
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
CHECKPOINT_DIR = 'checkpoints/'
CHECKPOINT_SAVE_PERIOD = 1

DATASET_BASE_PATH = '/mnt/e/CatFeeder/AI/TrainingData/cats/normalized/'

NOW = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

matplotlib.use('TkAgg')


def load_dataset(data_path: str):
    image_paths = []
    labels = []

    for image_path in tqdm.tqdm(glob.glob(f'{data_path}/*.jpg'), desc=f'Loading dataset from {os.path.basename(data_path)}'):
        # 根据图像路径构建预处理过的坐标文件路径
        coords_path = f"{image_path}.cat"

        # 直接读取预处理过并归一化的坐标
        with open(coords_path, 'r') as f:
            coords = list(map(float, f.readline().split()))

        labels.append(coords)
        image_paths.append(image_path)

    labels = np.array(labels, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, IMAGE_SIZE)
        return image_resized / 255.0, label

    dataset = dataset.map(
        parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache().shuffle(len(image_paths)).batch(
        BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset


def train(model: keras.models.Model):
    train_dataset = load_dataset(os.path.join(DATASET_BASE_PATH, 'train'))
    val_dataset = load_dataset(os.path.join(DATASET_BASE_PATH, 'val'))

    log_dir = 'logs/fit/' + NOW
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, NOW, 'cp-{epoch:04d}.ckpt')
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_best_only=False,
        period=CHECKPOINT_SAVE_PERIOD
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        restore_best_weights=True,
        mode='min',  # 由于我们的指标是loss，所以我们选择min模式
    )

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        min_delta=0.001,
        mode='min',
        verbose=1
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='mean_squared_error',
                  metrics=['mae']
                  )

    model.fit(train_dataset,
              validation_data=val_dataset,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=[cp_callback, tensorboard_callback,
                         early_stopping_callback, reduce_lr_callback],
              )

    model.save(f'final-{NOW}.keras')


def evaluate(model: keras.models.Model):
    test_dataset = load_dataset(os.path.join(DATASET_BASE_PATH, 'test'))
    loss, mae = model.evaluate(test_dataset)
    print(f'Test loss: {loss}, Test MAE: {mae}')

    # 仅选择测试数据集的一个batch进行可视化
    for images, labels in test_dataset.take(1):
        predictions = model.predict(images)

        # 将预测的坐标反归一化，以便在原图上绘制
        predictions = predictions * \
            np.array([IMAGE_SIZE[0], IMAGE_SIZE[1]]
                     * (predictions.shape[1] // 2))
        labels = labels.numpy() * \
            np.array([IMAGE_SIZE[0], IMAGE_SIZE[1]] * (labels.shape[1] // 2))

        # 选择16张图像进行展示，如果batch大小小于16，则展示整个batch
        for i in range(min(16, images.shape[0])):
            img = images[i].numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换颜色空间以适配OpenCV的显示
            img = (img * 255).astype(np.uint8)  # 反归一化图像

            # 绘制真实坐标点
            for x, y in labels[i].reshape(-1, 2):
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

            # 绘制预测坐标点
            for x, y in predictions[i].reshape(-1, 2):
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

            # 显示图像
            plt.figure(figsize=(5, 5))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'Test Image {i+1}')
            plt.axis('off')
            plt.show()


def main():
    print(f'Using GPU: {len(tf.config.list_physical_devices("GPU")) > 0}')

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    model = MyCNNFeatureExtractionModel()

    train(model)

    model = keras.models.load_model(f'final-{NOW}.keras')
    evaluate(model)


if __name__ == "__main__":
    main()
