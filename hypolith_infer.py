"""
eureka_infer.py
Zhiang Chen, Jan 6 2020
eureka data inference
"""

import transforms as T
from engine import train_one_epoch, evaluate
import utils
import torch
from hypolith import Dataset
from model import get_rock_model_instance_segmentation

import os
from shutil import copyfile
import pickle
import numpy as np
from model import visualize_gt
from model import visualize_pred
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

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

def test_performance(model, data, device, path):

    model.load_state_dict(torch.load(path + "/epoch_0039.param"))
    evaluate(model, data, device=device)

    model.load_state_dict(torch.load(path + "/epoch_0033.param"))
    evaluate(model, data, device=device)

    model.load_state_dict(torch.load(path + "/epoch_0030.param"))
    evaluate(model, data, device=device)

    model.load_state_dict(torch.load(path + "/epoch_0025.param"))
    evaluate(model, data, device=device)

def get_mean_std(input_channel, image_mean, image_std):
    if input_channel == 8:
        return image_mean, image_std
    elif input_channel == 3:
        return image_mean[:3], image_std[:3]
    elif input_channel == 5:
        return image_mean[:5], image_std[:5]
    elif input_channel == 6:
        return image_mean[:3] + image_mean[-3:], image_std[:3] + image_mean[-3:]
    elif input_channel == 4:
        return image_mean[:3] + [np.mean(image_mean[-3:]).tolist()], image_std[:3] + [np.mean(image_std[-3:]).tolist()]
    elif input_channel == 'dem':
        return image_mean[-3:], image_std[-3:]


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0') # or 'cuda:1' to choose GPU

    # our dataset has three classes only - background, non-damaged, and damaged
    num_classes = 2

    input_c = 3
    dataset_infer = Dataset("./datasets/hypolith/rgb_masks_test/", transforms=get_transform(train=False), include_name=True, input_channel=input_c)
    #dataset_test = Dataset("./datasets/C3_test/all_rocks/", transforms=get_transform(train=False), include_name=False, input_channel=input_c)
    #dataset = Dataset("./datasets/C3/aug/", transforms=get_transform(train=True), input_channel=input_c)
    image_mean, image_std, _, _ = dataset_infer.imageStat()
    #image_mean = [0.44413754410888107, 0.5006070146982871, 0.5535905318603589]
    #image_std = [0.19215971846127658, 0.19749652155340464, 0.1958481473755369]
    image_mean, image_std = get_mean_std(input_c, image_mean, image_std)

    mask_rcnn = get_rock_model_instance_segmentation(num_classes, input_channel=input_c, image_mean=image_mean, image_std=image_std)
    # move model to the right device
    mask_rcnn.to(device)

    mask_rcnn.eval()

    instances = []

    mask_rcnn.load_state_dict(torch.load("trained_param_hypolith/epoch_0008.param"))

    f = 0

    for i, data in enumerate(dataset_infer):
        print(i)
        image, target = data
        pred = mask_rcnn(image.unsqueeze(0).to(device))[0]

        boxes = pred['boxes'].to("cpu").detach().numpy()
        labels = pred['labels'].to("cpu").detach().numpy()
        scores = pred['scores'].to("cpu").detach().numpy()
        masks = pred['masks'].to("cpu").detach().numpy()
        image_name = target['image_name']

        result = {}
        result['bb'] = boxes
        result['labels'] = labels
        result['scores'] = scores
        result['mask'] = masks
        result['image_name'] = image_name
        instances.append(result)
        #result['coord'] = [int(i)*390 for i in image_name.split('/')[-1].split('.')[0].split('_')]

        nm = masks.shape[0]
        for i in range(nm):
            rock = {}
            rock['bb'] = boxes[i]
            rock['mask'] = masks[i, 0, :, :]
            rock['score'] = scores[i]
            #rock['coord'] = [int(i)*390 for i in image_name.split('/')[-1].split('.')[0].split('_')
            #instances.append(rock)

        true_mask = visualize_gt(image, target, display=False)
        pred_mask = visualize_pred(image, pred, thred=0.7, display=False)
        name = image_name.split('/')[-1][:-4]+".jpg"
        save_path = "datasets/hypolith/results/"
        print(name)
        print(image_name)
        cv2.imwrite(save_path+"pred_"+name, pred_mask)
        cv2.imwrite(save_path+"true_"+name, true_mask)
        if len(instances) >= 30000:
            name = "./hypolith_%02d.pickle" % f
            f += 1
            with open(name, 'wb') as filehandle:
                pickle.dump(instances, filehandle)
                instances = []

    name = "./hypolith_%02d.pickle" % f
    with open(name, 'wb') as filehandle:
        pickle.dump(instances, filehandle)

