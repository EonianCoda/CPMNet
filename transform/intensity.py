# -*- coding: utf-8 -*-
from __future__ import print_function, division

from .abstract_transform import AbstractTransform
from scipy import ndimage
from skimage.util import random_noise
import random
import numpy as np


class RandomBlur(AbstractTransform):
    """
    Randomly applies Gaussian blur to the input image.

    Args:
        sigma_range (tuple): Range of sigma values for Gaussian blur. Default is (0.4, 0.8).
        p (float): Probability of applying the transform. Default is 0.5.
        channel_apply (int): Index of the channel to apply the transform on. Default is 0.

    Returns:
        dict: Transformed sample with the blurred image.

    """
    def __init__(self, sigma_range=(0.4, 0.8), p=0.5):
        """
        Initializes the RandomBlur transform.

        Args:
            sigma_range (tuple): Range of sigma values for Gaussian blur. Default is (0.4, 0.8).
            p (float): Probability of applying the transform. Default is 0.5.
        """
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, sample):
        """
        Applies the RandomBlur transform to the input sample.

        Args:
            sample (dict): Input sample containing the image.

        Returns:
            dict: Transformed sample with the blurred image.

        """
        if random.random() < self.p:
            image = sample['image']
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            if len(sample['image'].shape) == 3: # depth, height, width
                image_t = ndimage.gaussian_filter(image, sigma)
                sample['image'] = image_t
            elif len(sample['image'].shape) == 4:
                image_t = ndimage.gaussian_filter(image[0], sigma)
                sample['image'][0] = image_t

        return sample
    
class RandomBlurNodule(AbstractTransform):
    """
    Randomly applies Gaussian blur to the input image.

    Args:
        sigma_range (tuple): Range of sigma values for Gaussian blur. Default is (0.4, 0.8).
        p (float): Probability of applying the transform. Default is 0.5.
        channel_apply (int): Index of the channel to apply the transform on. Default is 0.

    Returns:
        dict: Transformed sample with the blurred image.

    """
    def __init__(self, sigma_range=(0.4, 0.6), p=0.5, offset = 1):
        """
        Initializes the RandomBlur transform.

        Args:
            sigma_range (tuple): Range of sigma values for Gaussian blur. Default is (0.4, 0.8).
            p (float): Probability of applying the transform. Default is 0.5.
        """
        self.sigma_range = sigma_range
        self.p = p
        self.offset = offset

    def __call__(self, sample):
        """
        Applies the RandomBlur transform to the input sample.

        Args:
            sample (dict): Input sample containing the image.

        Returns:
            dict: Transformed sample with the blurred image.

        """
        if random.random() < self.p and len(sample['ctr']) > 0:
            image = sample['image']
            ctr = sample['ctr'] # shape: [n, 3]
            rad = sample['rad'] # shape: [n, 3]
            
            # [z1, y1, x1, z2, y2, x2] = [ctr - rad / 2, ctr + rad / 2
            bboxes = np.array([ctr - rad / 2, ctr + rad / 2]).transpose(1, 0, 2).reshape(-1, 6)
            for box in bboxes:
                sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
                box[:3] = np.maximum(np.floor(box[:3] - self.offset), 0)
                box[3:] = np.minimum(np.ceil(box[3:] + self.offset), image.shape[1:])
                box = box.astype(np.int32)
                z1, y1, x1, z2, y2, x2 = box
                image[0, z1:z2, y1:y2, x1:x2] = ndimage.gaussian_filter(image[0, z1:z2, y1:y2, x1:x2], sigma)    

        return sample


class RandomGamma(AbstractTransform):
    """

    """

    def __init__(self, gamma_range=2, p=0.5, channel_apply=0):
        """
        gamma range: gamme will be in [1/gamma_range, gamma_range]
        """
        self.gamma_range = gamma_range
        self.channel_apply = channel_apply
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            gamma = np.random.uniform(1, self.gamma_range)
            if random.random() < 0.5:
                gamma = 1. / gamma
            image_t = np.power(image[self.channel_apply], gamma)
            image[self.channel_apply] = image_t
            sample['image'] = image

        return sample


class RandomNoise(AbstractTransform):
    def __init__(self, p=0.5, gamma_range=(1e-4, 5e-4)):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            if len(sample['image'].shape) == 3: # depth, height, width
                image_t = random_noise(image, var=gamma)
                sample['image'] = image_t
            elif len(sample['image'].shape) == 4: # channel, depth, height, width
                image_t = random_noise(image[0], var=gamma)
                sample['image'][0] = image_t

        return sample
