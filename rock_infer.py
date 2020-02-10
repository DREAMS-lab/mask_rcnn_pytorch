"""
eureka_infer.py
Zhiang Chen, Jan 6 2020
eureka data inference
"""

import transforms as T
from engine import train_one_epoch, evaluate
import utils
import torch
from rock import Dataset
from model import get_rock_model_instance_segmentation

import os
from shutil import copyfile
import pickle
import numpy as np
from model import visualize_result
from model import visualize_pred

class ToTensor(object):
    def __call__(self, image, target):
        # image = F.to_tensor(image).float()
        image = torch.from_numpy(image / 255.0).float()
        image = image.permute((2, 0, 1))
        return image, target

def get_transform(train):
    transforms = []
    transforms.append(ToTensor()) # torchvision.transforms.functional is a garbage, sorry guys
    return T.Compose(transforms)

if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0')

    # our dataset has three classes only - background, non-damaged, and damaged
    num_classes = 2

    dataset_test = Dataset("./datasets/Rock/mult/", transforms=get_transform(train=False), training=False, input_channel=8)
    dataset = Dataset("./datasets/Rock/data/", transforms=get_transform(train=True), input_channel=8)
    image_mean, image_std, _, _ = dataset.imageStat()
    #image_mean = image_mean[:3] #+ image_mean[-3:]
    #image_std = image_std[:3] #+ image_std[-3:]

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    mask_rcnn = get_rock_model_instance_segmentation(num_classes, input_channel=8, image_mean=image_mean, image_std=image_std)

    mask_rcnn.load_state_dict(torch.load("trained_param_8/epoch_0045.param"))

    # move model to the right device
    mask_rcnn.to(device)

    mask_rcnn.eval()

    for data in dataset_test:
        image, target = data
        pred = mask_rcnn(image.unsqueeze(0).to(device))[0]

        boxes = pred['boxes'].to("cpu").detach().numpy()
        labels = pred['labels'].to("cpu").detach().numpy()
        scores = pred['scores'].to("cpu").detach().numpy()
        masks = pred['masks'].to("cpu").detach().numpy()
        image_name = target['image_name']
        result = {}
        result['boxes'] = boxes
        result['labels'] = labels
        result['scores'] = scores
        result['masks'] = masks
        result['image_name'] = image_name

        # visualize_result(mask_rcnn, data)
        visualize_pred(image, pred)

        #with open('./datasets/Eureka_infer/104_pred/' + image_name.split('/')[-1][:-4] + ".pickle", 'wb') as filehandle:
        #    pickle.dump(result, filehandle)

