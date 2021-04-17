"""
eureka_infer.py
Zhiang Chen, Jan 6 2020
eureka data inference
"""

import transforms as T
from engine import train_one_epoch, evaluate
import utils
import torch
from data import Dataset
from model import get_model_instance_segmentation

import os
from shutil import copyfile
import pickle

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def fake_annotations(image_path, label_path):
    """
    there must be some samples in label_path such that the function can copy them and create fake annotations.
    :param image_path:
    :param label_path:
    :return:
    """
    image_files = os.listdir(image_path)
    label_files = os.listdir(label_path)
    cls_files = [f for f in label_files if f.endswith("cls.npy")]
    mask_files = [f for f in label_files if f.endswith("nd.npy")]
    cls_source = os.path.join(label_path, cls_files[0])
    mask_source = os.path.join(label_path, mask_files[0])
    for image_file in image_files:
        id = image_file[:-4]
        cls_file = os.path.join(label_path, id + "_cls.npy")
        mask_file = os.path.join(label_path, id + "_nd.npy")
        copyfile(cls_source, cls_file)
        copyfile(mask_source, mask_file)

if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has three classes only - background, non-damaged, and damaged
    num_classes = 6  # 3 or 6

    dataset_test = Dataset("./datasets/Eureka_infer/102/", "./datasets/Eureka_infer/102_labels/", get_transform(train=False), readsave=False)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    mask_rcnn = get_model_instance_segmentation(num_classes, image_mean=None, image_std=None, stats=False)

    mask_rcnn.load_state_dict(torch.load("trained_param_eureka_aug_mult/epoch_0021.param"))
    print("loaded weights")

    # move model to the right device
    mask_rcnn.to(device)

    mask_rcnn.eval()

    for image, target in dataset_test:
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

        with open('./datasets/Eureka_infer/102_pred/mult_aug/' + image_name.split('/')[-1][:-4] + ".pickle", 'wb') as filehandle:
            pickle.dump(result, filehandle)


