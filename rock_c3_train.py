"""
eureka_train.py
Zhiang Chen, Feb 2020
train mask rcnn with eureka data
"""

import transforms as T
from engine import train_one_epoch, evaluate
import utils
import torch
from rock_c3 import Dataset
from model import get_rock_model_instance_segmentation
import numpy as np
import time

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

def get_mean_std(input_channel, image_mean, image_std):
    if input_channel == 3:
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

    input_c = 4
    # use our dataset and defined transformations
    dataset = Dataset("./datasets/C3/aug/", transforms=get_transform(train=True), include_name=False, input_channel=input_c)
    dataset_test = Dataset("./datasets/C3_test/rocks/", transforms=get_transform(train=False), include_name=False, input_channel=input_c)
    image_mean, image_std, _, _ = dataset.imageStat()

    #image_mean = [0.23924888725523394, 0.2180423480395164, 0.2118836715688813, 0.26721142156890876, 0.32996910784324385, 0.1461123186277879, 0.5308107499991753, 0.28652559313771186]
    #image_std = [0.1459739643338365, 0.1311105424825076, 0.12715888419418298, 0.149469170605332, 0.15553466224696225, 0.10533129832132752, 0.24088403135495345, 0.24318892151508417]

    image_mean, image_std = get_mean_std(input_c, image_mean, image_std)

    # split the dataset in train and test set
    #indices = torch.randperm(len(dataset)).tolist()
    #dataset = torch.utils.data.Subset(dataset, indices)
    #indices_test = torch.randperm(len(dataset_test)).tolist()
    #dataset_test = torch.utils.data.Subset(dataset_test, indices_test)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=8,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    # get the model using our helper function
    mask_rcnn = get_rock_model_instance_segmentation(num_classes, input_channel=input_c, image_mean=image_mean, image_std=image_std)

    read_param = False
    if read_param:
        mask_rcnn.load_state_dict(torch.load("trained_param_6/epoch_0050.param"))
        print("Loaded weights")

    # move model to the right device
    mask_rcnn.to(device)

    # construct an optimizer
    params = [p for p in mask_rcnn.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=8,
                                                   gamma=0.1)
    init_epoch = 0
    num_epochs = 20

    for epoch in range(init_epoch, init_epoch + num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(mask_rcnn, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(mask_rcnn, data_loader_test, device=device)

        save_param = "trained_param_c3_4/epoch_{:04d}.param".format(epoch)
        torch.save(mask_rcnn.state_dict(), save_param)
