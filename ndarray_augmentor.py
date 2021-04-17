import cv2
import numpy as np
import os
import uuid
import copy


def rotateImage(image, angle):
    l = len(image.shape)
    image_center = tuple(np.array(image.shape[:2]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
    if len(result.shape) < l:
        y, x = result.shape
        result = result.reshape((y, x, 1))
    return result


def zoom(image, zoom_scale):
    size = image.shape
    l = len(size)
    image = cv2.resize(image, None, fx=zoom_scale, fy=zoom_scale)
    if len(image.shape) < l:
        y, x = image.shape
        image = image.reshape((y, x, 1))
    new_size = image.shape

    if len(size) == 3:
        if zoom_scale > 1:
            return image[int((new_size[0] - size[0]) / 2): int((new_size[0] - size[0]) / 2 + size[0]),
                   int((new_size[1] - size[1]) / 2): int((new_size[1] - size[1]) / 2 + size[1]), :]
        elif zoom_scale == 1:
            return image
        else:
            new_image = np.zeros(size).astype('uint8')
            new_image[int((size[0] - new_size[0]) / 2): int((size[0] - new_size[0]) / 2 + new_size[0]),
            int((size[1] - new_size[1]) / 2): int((size[1] - new_size[1]) / 2 + new_size[1]), :] = image
            return new_image


def sample(image, rotation_min, rotation_max, fliplr, flipud, zoom_min, zoom_max):
    angle = np.random.uniform(rotation_min, rotation_max)
    image = rotateImage(image, angle)

    if fliplr:
        if np.random.random() < 0.5:
            image = np.fliplr(image)

    if flipud:
        if np.random.random() < 0.5:
            image = np.flipud(image)

    zoom_scale = np.random.uniform(zoom_min, zoom_max)
    image = zoom(image, zoom_scale)

    return image

def augmentor(npy_path, save_path, batch_number=1, rotation_min=0, rotation_max=0, fliplr=False, flipud=False, zoom_min=1, zoom_max=1, input_c=6):
    c = 0
    npy_files = [os.path.join(npy_path, f) for f in os.listdir(npy_path) if f.endswith('.npy')]
    while c < batch_number:
        for npy_file in npy_files:
            data = np.load(npy_file)
            data = sample(data, rotation_min, rotation_max, fliplr, flipud, zoom_min, zoom_max)
            unid = uuid.uuid4().hex
            image = data[:, :, :input_c]
            masks = data[:, :, input_c:]
            num_objs = masks.shape[2]
            print(masks.shape[2])
            for i in reversed(range(num_objs)):
                mask = masks[:, :, i]
                if mask.max() < 250:
                    masks = np.delete(masks, i, axis=2)
                    continue
                mask = mask >= 250
                pos = np.where(mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if xmin >= xmax:
                    masks = np.delete(masks, i, axis=2)
                    continue
                if ymin >= ymax:
                    masks = np.delete(masks, i, axis=2)
                    continue
            print(masks.shape[2])
            print('\n')
            data = np.append(image, masks, axis=2)
            save_file = npy_file.split('.npy')[0].split('/')[-1] + "_" + unid + ".npy"
            save_file = os.path.join(save_path, save_file)
            np.save(save_file, data)

        c += 1

def balanced_augmentor(image_path, label_path, aug_path, augmentation_batch=1, augmentation_ratio=[],
                       rotation_min=0, rotation_max=0, fliplr=False, flipud=False, zoom_min=1, zoom_max=1):

    image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg')]
    while augmentation_batch:
        augmentation_batch -= 1
        for image_file in image_files:
            f_name = image_file.split('/')[-1][:-4]
            mask_file = os.path.join(label_path, f_name + "_nd.npy")
            cls_file = os.path.join(label_path, f_name + "_cls.npy")

            if not os.path.isfile(mask_file):
                continue
            if not os.path.isfile(cls_file):
                continue

            image = cv2.imread(image_file)
            masks = np.load(mask_file)
            clses = np.load(cls_file)  # cls should start with 0, e.g. [0, 1, 2, 3, ...]
            image_mask = np.concatenate((image, masks), axis=2)

            n = augmentation_ratio[int(clses.max())]
            for _ in range(n):
                data = copy.deepcopy(image_mask)
                data = sample(data, rotation_min, rotation_max, fliplr, flipud, zoom_min, zoom_max)
                image = data[:, :, :3]
                masks = data[:, :, 3:]
                cls = copy.deepcopy(clses)

                if masks.max() < 0.1:
                    continue

                print(masks.shape[2])
                num_objs = masks.shape[2]
                for i in reversed(range(num_objs)):
                    mask = masks[:, :, i]
                    if mask.max() < 250:
                        masks = np.delete(masks, i, axis=2)
                        cls = np.delete(cls, i)
                        continue

                    mask = mask >= 250
                    pos = np.where(mask)
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])

                    if (xmin >= xmax) | (ymin >= ymax):
                        masks = np.delete(masks, i, axis=2)
                        cls = np.delete(cls, i)
                        continue

                if masks.shape[2] == 0:
                    continue

                if masks.max() == 0:
                    continue

                print(masks.shape[2])
                print('\n')
                unid = uuid.uuid4().hex

                new_mask_file = os.path.join(aug_path, f_name + "_" + unid + "_nd.npy")
                new_cls_file = os.path.join(aug_path, f_name + "_" + unid + "_cls.npy")
                new_image_file = os.path.join(aug_path, f_name + "_" + unid + ".jpg")

                np.save(new_mask_file, masks)
                np.save(new_cls_file, cls)
                cv2.imwrite(new_image_file, image)


if __name__ == '__main__':
    config = dict(
        batch_number=6,
        rotation_min=0,#-90,
        rotation_max=0,#90,
        fliplr=True,
        flipud=True,
        zoom_min=1,#0.8,
        zoom_max=1,#1.2,
        input_c=6)

    config_ = dict(
        augmentation_batch=5,
        augmentation_ratio=[1, 3, 15, 50, 100],
        rotation_min=-90,
        rotation_max=90,
        fliplr=True,
        flipud=True,
        zoom_min=0.8,
        zoom_max=1.2)
    #image_path = './datasets/Eureka/images/'
    #label_path = './datasets/Eureka/labels/'
    #aug_path = './datasets/Eureka/aug/'
    #balanced_augmentor(image_path, label_path, aug_path, **config_)

    npy_path = './datasets/hypolith/rgb_masks/'
    aug_path = './datasets/hypolith/aug/'
    augmentor(npy_path, aug_path, **config)
