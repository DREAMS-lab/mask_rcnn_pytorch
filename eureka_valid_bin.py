"""
eureka_train.py
Zhiang Chen, Dec 25 2019
train mask rcnn with eureka data
"""

import transforms as T
from engine import train_one_epoch, evaluate
import utils
import torch
from data import Dataset
from model import get_model_instance_segmentation
import os

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0')

    # our dataset has three classes only - background, non-damaged, and damaged
    # background, nd, d0, d1, d2, d3
    num_classes = 3 # 3 or 6
    # use our dataset and defined transformations
    dataset = Dataset("./datasets/Eureka/images/", "./datasets/Eureka/labels/", get_transform(train=True), readsave=False, include_name=False, binary_cls=True)
    dataset_test = Dataset("./datasets/Eureka/images_test/", "./datasets/Eureka/labels/", get_transform(train=False), savePickle=False,readsave=False, include_name=False, binary_cls=True)
    print(len(dataset_test))
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
    indices_test = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    mask_rcnn = get_model_instance_segmentation(num_classes, None, None, False)

    read_param = False
    if read_param:
        mask_rcnn.load_state_dict(torch.load("trained_param_eureka_aug_bin/epoch_0009.param"))

    # move model to the right device
    mask_rcnn.to(device)

    # construct an optimizer
    params = [p for p in mask_rcnn.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.00001)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=10,
                                                   gamma=0.1)

    init_epoch = 0
    num_epochs = 32

    #save_param = "trained_param_eureka_mult/epoch_{:04d}.param".format(0)
    #torch.save(mask_rcnn.state_dict(), save_param)
    weight_path = "trained_param_eureka_bin"
    weights = [f for f in os.listdir(weight_path) if f.endswith(".param")]
    weights.sort()
    mask_rcnn.load_state_dict(torch.load("trained_param_eureka_bin/epoch_0019.param"))
    evaluate(mask_rcnn, data_loader_test, device=device)

    """
    for weight in weights:
        weight_name = os.path.join(weight_path, weight)
        mask_rcnn.load_state_dict(torch.load(weight_name))
        print(weight_name)
        #save_param = "trained_param_eureka_mult/epoch_{:04d}.param".format(epoch)
        #torch.save(mask_rcnn.state_dict(), save_param)

        #save_param = "trained_param_eureka_aug_bin/epoch_{:04d}.param".format(epoch)
        #torch.save(mask_rcnn.state_dict(), save_param)

        # train for one epoch, printing every 10 iterations
        #train_one_epoch(mask_rcnn, optimizer, data_loader, device, epoch, print_freq=500)
        # update the learning rate
        #lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(mask_rcnn, data_loader_test, device=device)

        #torch.save(mask_rcnn.state_dict(), save_param)
    """

"""
num_classes = 3
# use our dataset and defined transformations
dataset = Dataset("./datasets/Eureka/images/", "./datasets/Eureka/labels/", get_transform(train=True), readsave=False)
dataset_test = Dataset("./datasets/Eureka/images_test/", "./datasets/Eureka/labels_test/", get_transform(train=False), readsave=False)

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices)
indices_test = torch.randperm(len(dataset_test)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices_test)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
mask_rcnn = get_model_instance_segmentation(num_classes)

read_param = True
if read_param:
    mask_rcnn.load_state_dict(torch.load("trained_param/epoch_0099.param"))

# move model to the right device
mask_rcnn.to(device)
"""
