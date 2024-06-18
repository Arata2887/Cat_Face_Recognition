import os
import cv2
import tqdm
import numpy as np
import glob

# 定义原始数据和保存数据的路径
ORIGINAL_DATASET_PATH = '/mnt/e/CatFeeder/AI/TrainingData/cats/val/'
PROCESSED_DATASET_PATH = '/mnt/e/CatFeeder/AI/TrainingData/cats/normalized/val/'
IMAGE_SIZE = (224, 224)

# 确保处理后的数据目录存在
os.makedirs(PROCESSED_DATASET_PATH, exist_ok=True)

def normalize_and_resize(image_path, coords):
    # 读取图像
    image = cv2.imread(image_path)
    original_size = image.shape[:2]  # (height, width)

    # 归一化坐标
    normalized_coords = np.array(coords, dtype=np.float32)
    normalized_coords[0::2] /= original_size[1]  # x坐标归一化
    normalized_coords[1::2] /= original_size[0]  # y坐标归一化

    # 调整图像大小
    resized_image = cv2.resize(image, IMAGE_SIZE)

    return resized_image, normalized_coords

def process_dataset(dataset_path):
    image_paths = glob.glob(os.path.join(dataset_path, '*.jpg'))
    for image_path in tqdm.tqdm(image_paths, desc='Processing images'):
        # 从图像路径构建坐标文件路径
        coords_path = image_path + '.cat'

        # 读取并解析坐标
        with open(coords_path, 'r') as f:
            coords = list(map(int, f.readline().split()))[1:]  # 跳过第一个数字

        # 归一化坐标并调整图像大小
        processed_image, normalized_coords = normalize_and_resize(image_path, coords)

        # 构建保存路径
        save_image_path = image_path.replace(ORIGINAL_DATASET_PATH, PROCESSED_DATASET_PATH)
        save_coords_path = coords_path.replace(ORIGINAL_DATASET_PATH, PROCESSED_DATASET_PATH)

        # 保存处理后的图像和坐标
        os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
        cv2.imwrite(save_image_path, processed_image)
        with open(save_coords_path, 'w') as f:
            f.write(' '.join(map(str, normalized_coords.tolist())))

def test():
    import random
    # 从处理后的数据集中随机选择16张图像
    image_paths = glob.glob(os.path.join(PROCESSED_DATASET_PATH, '*.jpg'))
    image_paths = random.sample(image_paths, 16)

    # 读取图像和坐标
    images = [cv2.imread(image_path) for image_path in image_paths]
    coords = [np.loadtxt(image_path + '.cat', dtype=np.float32) for image_path in image_paths]

    # 显示图像和坐标
    for i in range(16):
        image = images[i]
        coord = coords[i]

        # 将坐标还原到原始图像大小
        original_size = image.shape[:2]
        coord[0::2] *= original_size[1]  # x坐标还原
        coord[1::2] *= original_size[0]  # y坐标还原

        # 绘制坐标
        for j in range(0, len(coord), 2):
            cv2.circle(image, (int(coord[j]), int(coord[j + 1])), 3, (0, 255, 0), -1)

        # 显示图像
        cv2.imshow(f'Image {image_paths[i]}', image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_dataset(ORIGINAL_DATASET_PATH)

    test()