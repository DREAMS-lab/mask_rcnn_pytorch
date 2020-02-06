import cv2
import numpy as np
import os
import uuid


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

def augmentor(npy_path, batch_number=1, rotation_min=0, rotation_max=0, fliplr=False, flipud=False, zoom_min=1, zoom_max=1):
    c = 0
    npy_files = [os.path.join(npy_path, f) for f in os.listdir(npy_path)]
    while c < batch_number:
        for npy_file in npy_files:
            data = np.load(npy_file)
            data = sample(data, rotation_min, rotation_max, fliplr, flipud, zoom_min, zoom_max)
            unid = uuid.uuid4().hex
            image = data[:, :, :8]
            masks = data[:, :, 8:]
            num_objs = masks.shape[2]
            for i in reversed(range(num_objs)):
                mask = masks[:, :, i]
                if mask.max() < 250:
                    masks = np.delete(masks, i, axis=2)
            data = np.append(image, masks, axis=2)
            save_file = npy_file.split('.npy')[0] + "_" + unid + ".npy"
            np.save(save_file, data)

        c += 1

if __name__ == '__main__':
    config = dict(
        batch_number=2,
        rotation_min=-90,
        rotation_max=90,
        fliplr=True,
        flipud=True,
        zoom_min=0.8,
        zoom_max=1.2)

    npy_path = './datasets/Rock/data/'
    augmentor(npy_path, **config)