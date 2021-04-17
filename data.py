"""
data.py
Zhiang Chen, Dec 24 2019
data class for mask rcnn
"""

import os
import numpy as np
import torch
from PIL import Image
import pickle

"""
./datasets/
    Eureka/
        images/
            Eureka_101_0_1.jpg
            ...
        labels/
            Eureka_101_0_1_cls.npy
            Eureka_101_0_1_nd.npy
            ...
"""

class Dataset(object):
    def __init__(self, image_path, label_path, transforms=None, savePickle=True, readsave=True, include_name=True, binary_cls=False):
        self.image_path = image_path
        self.label_path = label_path
        self.transforms = transforms
        self.images = [f for f in os.listdir(image_path) if f.endswith(".jpg")]
        self.masks = [f for f in os.listdir(label_path) if f.endswith("nd.npy")]
        self.classes = [f for f in os.listdir(label_path) if f.endswith("cls.npy")]
        self.include_name = include_name
        self.savePickle = savePickle
        self.__refine(readsave)
        self.binary_cls = binary_cls


    def __refine(self, readsave):
        """
        1. only keep image-mask-class matched files
        2. only keep files with roofs
        3. sort files
        """
        if not readsave:
            images = []
            masks = []
            classes = []
            for img in self.images:
                frame = img[:-4]
                mask = frame + "_nd.npy"
                cls = frame + "_cls.npy"
                if mask in self.masks:
                    if cls in self.classes:
                        mask_path = os.path.join(self.label_path, mask)
                        mask_nd = np.load(mask_path)
                        if mask_nd.max() > 0:
                            images.append(img)
                            masks.append(mask)
                            classes.append(cls)

            self.images = images
            self.masks = masks
            self.classes = classes
            data = {"images": images, "masks": masks, "classes": classes}
            if self.savePickle:
                with open('data.pickle', 'wb') as filehandle:
                    pickle.dump(data, filehandle)
        else:
            with open('data.pickle', 'rb') as filehandle:
                data = pickle.load(filehandle)
                self.images = data["images"]
                self.masks = data["masks"]
                self.classes = data["classes"]


    def __getitem(self, idx):
        img_path = os.path.join(self.image_path, self.images[idx])
        mask_path = os.path.join(self.label_path, self.masks[idx])
        cls_path = os.path.join(self.label_path, self.classes[idx])

        image = Image.open(img_path).convert("RGB")
        image = image.resize((1000, 1000))
        # 0 encoding non-damaged is supposed to be 1 for training.
        # In training, 0 is of background
        obj_ids = np.load(cls_path)
        masks = np.load(mask_path)
        masks = masks > 0  # convert to binary masks
        masks = masks.astype(np.uint8)

        num_objs = obj_ids.shape[-1]
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
        #labels = torch.ones((num_objs,), dtype=torch.int64)
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
            target["image_name"] = img_path

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __getitem__(self, idx):
        image, target = self.__getitem(idx)
        labels = target["labels"]
        if not self.binary_cls:
            labels = labels + 1 # multiple classes
        else:
            labels = (labels > 0).type(torch.int64) + 1  # only two classes, non-damaged and damaged
        target["labels"] = labels
        return image, target


    def __len__(self):
        return len(self.images)

    def display(self, idx):
        image, target = self.__getitem__(idx)
        #image = image.permute((1, 2, 0))
        #image = (image.numpy() * 255).astype(np.uint8)
        #image = Image.fromarray(image)
        image.show()

        masks = target["masks"]
        masks = masks.permute((1, 2, 0))
        masks = masks.numpy()
        masks = masks.max(axis=2) * 255
        masks = Image.fromarray(masks)
        masks.show()

if __name__  ==  "__main__":
    #ds = Dataset("./datasets/Eureka/images/", "./datasets/Eureka/labels/", readsave=True)
    ds = Dataset("./datasets/Eureka/aug/", "./datasets/Eureka/aug/", readsave=True)
    id = 0
    image, target = ds[id]
    image = np.array(image)
    ds.display(id)
    print(target['labels'])
