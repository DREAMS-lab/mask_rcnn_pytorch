"""
eureka_train.py
Zhiang Chen, Dec 25 2019
train mask rcnn with eureka data
"""

import transforms as T
from engine import train_one_epoch, evaluate
import utils
import torch
from c3 import Dataset
from model import get_rock_model_instance_segmentation
import numpy as np

torch.manual_seed(0)

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
'''
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
'''

def get_mean_std(input_channel, image_mean, image_std):
    if input_channel == 6:
        return image_mean, image_std
    elif input_channel == 3:
        return image_mean[:3], image_std[:3]
    elif input_channel == 4:
        return image_mean[:3] + [np.mean(image_mean[-3:]).tolist()], image_std[:3] + [np.mean(image_std[-3:]).tolist()]
    elif input_channel == -3:
        return image_mean[-3:], image_std[-3:]
    elif input_channel == 1:
        return [np.mean(image_mean[-3:]).tolist()], [np.mean(image_std[-3:]).tolist()]

if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0')

    # our dataset has three classes only - background, non-damaged, and damaged
    num_classes = 2

    input_c = 4
    # use our dataset and defined transformations
    dataset = Dataset("./datasets/iros/c3/aug/", transforms=get_transform(train=True), include_name=False, input_channel=input_c)
    ##dataset_test = Dataset("./datasets/Rock/data_test/", transforms=get_transform(train=False), include_name=False, input_channel=input_c)
    #dataset_test = Dataset("./datasets/Rock_test/mult/", transforms=get_transform(train=False), include_name=False, input_channel=input_c)
    dataset_test = Dataset("./datasets/iros/c3_test/rgbd_masks/", transforms=get_transform(train=False), include_name=False, input_channel=input_c)
    # image_mean, image_std, _, _ = dataset.imageStat()
    #image_mean = [0.2635908247051704, 0.2565450032962188, 0.24311759802366928, 0.3007502338171828, 0.35639093071269307, 0.24508291731782375]
    #image_std =  [0.23736204788409055, 0.23722317885795837, 0.22990199087409235, 0.25134661805921518, 0.26149370275679834, 0.26121346688112296]
    #image_mean = [0.5561083275961, 0.4762376069094619, 0.4467060158591905, 0.10208016634218137, 0.5028495619538484, 0.5806285215840447]
    #image_std =  [0.23056259201528329, 0.21163784115481502, 0.19388422413947035, 0.0311074642376007, 0.18291436237313496, 0.20893878043377523]
    image_mean = [0.5620081496860253, 0.48239326529405263, 0.452555695791186, 0.10258982219688167, 0.5069069552967871, 0.5892437452304018]
    image_std =  [0.22786734424312816, 0.21037678690930353, 0.19248594128629354, 0.030205125607848885, 0.1780133951799179, 0.20065141490899788]

    image_mean, image_std = get_mean_std(input_c, image_mean, image_std)

    # split the dataset in train and test set
    #indices = torch.randperm(len(dataset)).tolist()
    #dataset = torch.utils.data.Subset(dataset, indices)
    #indices_test = torch.randperm(len(dataset_test)).tolist()
    #dataset_test = torch.utils.data.Subset(dataset_test, indices_test)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=8,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    # get the model using our helper function
    mask_rcnn = get_rock_model_instance_segmentation(num_classes, input_channel=input_c, image_mean=image_mean, image_std=image_std, pretrained=True)

    read_param = False
    if read_param:
        mask_rcnn.load_state_dict(torch.load("trained_param_6/epoch_0050.param"))
        print("Loaded weights")

    # move model to the right device
    mask_rcnn.to(device)

    # construct an optimizer
    params = [p for p in mask_rcnn.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.5)
    init_epoch = 0
    num_epochs = 12

    save_param = "trained_param_c3_tl_rgbd1/epoch_{:04d}.param".format(init_epoch)
    torch.save(mask_rcnn.state_dict(), save_param)

    #'''
    for epoch in range(init_epoch, init_epoch + num_epochs):
        save_param = "trained_param_c3_tl_rgbd1/epoch_{:04d}.param".format(epoch)
        #torch.save(mask_rcnn.state_dict(), save_param)
        # train for one epoch, printing every 10 iterations
        print(save_param)
        train_one_epoch(mask_rcnn, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #print('\n')
        #print("trained_param_4/epoch_00%02d.param" % epoch)
        #mask_rcnn.load_state_dict(torch.load("trained_param_4/epoch_00%02d.param" % epoch))
        evaluate(mask_rcnn, data_loader_test, device=device)

        #save_param = "trained_param_8_fresh/epoch_{:04d}.param".format(epoch)
        torch.save(mask_rcnn.state_dict(), save_param)
    '''

    for epoch in range(init_epoch, init_epoch + num_epochs):
        #save_param = "trained_param_3_fresh/epoch_{:04d}.param".format(epoch)
        #torch.save(mask_rcnn.state_dict(), save_param)
        # train for one epoch, printing every 10 iterations
        #train_one_epoch(mask_rcnn, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        #lr_scheduler.step()
        # evaluate on the test dataset
        print('\n')
        name = "trained_param_8/epoch_00%02d.param" % epoch
        print(name)
        mask_rcnn.load_state_dict(torch.load(name))
        evaluate(mask_rcnn, data_loader_test, device=device)

        #save_param = "trained_param_8_fresh/epoch_{:04d}.param".format(epoch)
        #torch.save(mask_rcnn.state_dict(), save_param)
    '''
