"""
rock.py
Zhiang Chen, Feb 2020
data class for mask rcnn
"""

import os
import numpy as np
import torch
from PIL import Image
import pickle
import matplotlib.pyplot as plt

"""
./datasets/
    Rock/
        data/
            0_8.npy
            0_9.npy
            1_4.npy
            ...
"""

class Dataset(object):
    def __init__(self, data_path, transforms=None, input_channel=6, include_name=True):
        self.data_path = data_path
        self.transforms = transforms
        self.data_files = [f for f in os.listdir(data_path) if f.endswith(".npy")]
        self.input_channel = input_channel
        self.include_name = include_name

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_path, self.data_files[idx])

        data = np.load(data_path)

        if self.input_channel == 6:
            image = data[:, :, :self.input_channel]
        elif self.input_channel == 3:
            image = data[:, :, :3]
        elif self.input_channel == 4:
            rgb = data[:, :, :3]
            dem = data[:, :, 3:]
            d = dem[:,:,0]*0.33 + dem[:,:,1]*0.33 + dem[:,:,2]*0.33
            image = np.append(rgb, np.expand_dims(d, axis=2), axis=2)

        if data.shape[2] == 6:
            masks = np.ones_like(image[:, :, :3]) * 255
        else:
            masks = data[:, :, 6:]
        num_objs = masks.shape[2]
        """
        for i in reversed(range(num_objs)):
            mask = masks[:, :, i]
            if mask.max() < 250:
                masks = np.delete(masks, i, axis=2)
        num_objs = masks.shape[2]
        """
        # 0 encoding non-damaged is supposed to be 1 for training.
        # In training, 0 is of background
        obj_ids = np.ones(num_objs)

        masks = masks >= 250  # convert to binary masks

        boxes = []

        for i in range(num_objs):
            pos = np.where(masks[:, :, i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        masks = masks.permute((2, 0, 1))

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.include_name:
            target["image_name"] = data_path

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.data_files)

    def show(self, idx):
        image, target = self.__getitem__(idx)
        rgb = image[:, :, :3].astype(np.uint8)
        rgb = Image.fromarray(rgb)
        rgb.show()
        masks = target["masks"]
        masks = masks.permute((1, 2, 0))
        masks = masks.numpy()
        masks = masks.max(axis=2) * 255
        masks = Image.fromarray(masks)
        masks.show()

    def imageStat(self):
        images = np.empty((0, 6), float)
        for data_file in self.data_files:
            if len(data_file.split('_'))==2:
                data_path = os.path.join(self.data_path, data_file)
                data = np.load(data_path)
                print(data.shape)
                image = data[:, :, :6].astype(float).reshape(-1, 6)/255.0
                images = np.append(images, image, axis=0)
        return np.mean(images, axis=0).tolist(), np.std(images, axis=0).tolist(), \
               np.max(images, axis=0).tolist(), np.min(images, axis=0).tolist()


    def imageStat2(self):
        images = np.empty((0, 3), float)
        import random
        random.shuffle(self.data_files)
        for data_file in self.data_files[:40]:
            if True:
                data_path = os.path.join(self.data_path, data_file)
                data = np.load(data_path)
                image = data[:, :, :3].astype(float).reshape(-1, 3)/255.0
                images = np.append(images, image, axis=0)
        return np.mean(images, axis=0).tolist(), np.std(images, axis=0).tolist(), \
               np.max(images, axis=0).tolist(), np.min(images, axis=0).tolist()


if __name__  ==  "__main__":
    #ds = Dataset("./datasets/Rock/data/")
    ds = Dataset("./datasets/hypolith_sample_set_throop/npy",input_channel=3)
    # image_mean, image_std, image_max, image_min = ds.imageStat()


    id = 29
    image, target = ds[id]
    print(target['image_name'])
    ds.show(id)

    id = 28
    image, target = ds[id]
    print(target['image_name'])
    ds.show(id)
    print(ds.imageStat2())
