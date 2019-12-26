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

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has three classes only - background, non-damaged, and damaged
    num_classes = 3
    # use our dataset and defined transformations
    dataset = Dataset("./datasets/Eureka/images/", "./datasets/Eureka/labels/", get_transform(train=True))
    dataset_test = Dataset("./datasets/Eureka/images/", "./datasets/Eureka/labels/", get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

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
        mask_rcnn.load_state_dict(torch.load("trained_param/epoch_0001.param"))

    # move model to the right device
    mask_rcnn.to(device)

    # construct an optimizer
    params = [p for p in mask_rcnn.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    init_epoch = 2
    num_epochs = 8

    for epoch in range(init_epoch, init_epoch + num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(mask_rcnn, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(mask_rcnn, data_loader_test, device=device)

        save_param = "trained_param/epoch_{:04d}.param".format(epoch)
        torch.save(mask_rcnn.state_dict(), save_param)


