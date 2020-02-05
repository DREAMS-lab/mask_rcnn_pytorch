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
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        self.data_files = [f for f in os.listdir(data_path) if f.endswith(".npy")]



    def __getitem__(self, idx):
        data_path = os.path.join(self.data_path, self.data_files[idx])

        data = np.load(data_path)
        image = data[:, :, :8]
        masks = data[:, :, 8:]
        num_objs = masks.shape[2]
        # 0 encoding non-damaged is supposed to be 1 for training.
        # In training, 0 is of background
        obj_ids = np.ones(num_objs)
        masks = masks == 255  # convert to binary masks

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

if __name__  ==  "__main__":
    ds = Dataset("./datasets/Rock/data/")
    image, target = ds[4]
    print(image)
    print(image.shape)
    print(target)
    ds.show(4)
