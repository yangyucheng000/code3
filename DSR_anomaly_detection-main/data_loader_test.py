import numpy as np
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
import os
import mindspore.ops
from mindspore import Tensor


class TestMVTecDataset():
    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir

        self.images = sorted(glob.glob(root_dir+"/*/*.png"))    # 理论上这里传入的应该是.../cls_name/test/
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)


    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)    # NumPy 数组，HWC，"BGR"
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # cv2.IMREAD_GRAYSCALE：加载为灰度图像
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            #h, w
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):     # question
            idx = idx.asnumpy().tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        # good, crack, scratch, etc.
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)


        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample
