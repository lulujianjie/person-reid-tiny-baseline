from PIL import Image
import numpy as np
import random
import math


class GaussianMask(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
        width = img.size[0]
        height = img.size[1]
        mask = np.zeros((height, width))
        mask_h = np.zeros((height, width))
        mask_h += np.arange(0, width) - width / 2
        mask_v = np.zeros((width, height))
        mask_v += np.arange(0, height) - height / 2
        mask_v = mask_v.T

        numerator = np.power(mask_h, 2) + np.power(mask_v, 2)
        denominator = 2 * (height * height + width * width)
        mask = np.exp(-(numerator / denominator))

        img = np.asarray(img)
        new_img = np.zeros_like(img)
        new_img[:, :, 0] = np.multiply(mask, img[:, :, 0])
        new_img[:, :, 1] = np.multiply(mask, img[:, :, 1])
        new_img[:, :, 2] = np.multiply(mask, img[:, :, 2])

        return Image.fromarray(new_img)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
