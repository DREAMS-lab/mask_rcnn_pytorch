"""
eureka_infer.py
Zhiang Chen, Jan 6 2020
eureka data inference
"""

import transforms as T
from engine import train_one_epoch, evaluate
import utils
import torch
from rock_c3 import Dataset
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
    device = torch.device('cuda:0')

    # our dataset has three classes only - background, non-damaged, and damaged
    num_classes = 2

    input_c = 3
    dataset_infer = Dataset("./datasets/C3/rgbd/", transforms=get_transform(train=False), include_name=True, input_channel=input_c)
    dataset_test = Dataset("./datasets/C3_test/all_rocks/", transforms=get_transform(train=False), include_name=False, input_channel=input_c)
    dataset = Dataset("./datasets/C3/aug/", transforms=get_transform(train=True), input_channel=input_c)
    image_mean, image_std, _, _ = dataset.imageStat()
    # image_mean = [0.23924888725523394, 0.2180423480395164, 0.2118836715688813, 0.26721142156890876, 0.32996910784324385, 0.1461123186277879, 0.5308107499991753, 0.28652559313771186]
    # image_std = [0.1459739643338365, 0.1311105424825076, 0.12715888419418298, 0.149469170605332, 0.15553466224696225, 0.10533129832132752, 0.24088403135495345, 0.24318892151508417]
    image_mean, image_std = get_mean_std(input_c, image_mean, image_std)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    mask_rcnn = get_rock_model_instance_segmentation(num_classes, input_channel=input_c, image_mean=image_mean, image_std=image_std)
    # move model to the right device
    mask_rcnn.to(device)

    mask_rcnn.eval()


    # test_performance(mask_rcnn, data_loader_test, device, "trained_param_8")

    instances = []

    '''
    init_epoch = 10
    num_epochs = 25
    
    for epoch in range(init_epoch, init_epoch + num_epochs):
        """
        # train for one epoch, printing every 10 iterations
        train_one_epoch(mask_rcnn, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()
        """
        print("trained_param_c3_4/epoch_00%02d.param" % epoch)
        mask_rcnn.load_state_dict(torch.load("trained_param_c3_4/epoch_00%02d.param" % epoch))
        # evaluate on the test dataset
        evaluate(mask_rcnn, data_loader_test, device=device)
    '''

    mask_rcnn.load_state_dict(torch.load("trained_param_c3_3/epoch_0018.param"))
    #mask_rcnn.load_state_dict(torch.load("trained_param_c3_3/epoch_0002.param"))

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
        result['coord'] = [int(i)*390 for i in image_name.split('/')[-1].split('.')[0].split('_')]

        nm = masks.shape[0]
        for i in range(nm):
            rock = {}
            rock['bb'] = boxes[i]
            rock['mask'] = masks[i, 0, :, :]
            rock['score'] = scores[i]
            rock['coord'] = [int(i)*390 for i in image_name.split('/')[-1].split('.')[0].split('_')]
            instances.append(rock)

        #visualize_result(mask_rcnn, data)
        #visualize_pred(image, pred)
        if len(instances) >= 30000:
            name = "./datasets/C3/rocks_c3_3_18_%02d.pickle" % f
            f += 1
            with open(name, 'wb') as filehandle:
                pickle.dump(instances, filehandle)
                instances = []

    name = "./datasets/C3/rocks_c3_3_18_%02d.pickle" % f
    with open(name, 'wb') as filehandle:
        pickle.dump(instances, filehandle)
    
    #'''

