from typing import List, Tuple
import numpy as np
import copy

class SplitComb():
    def __init__(self, crop_size: List[int] = [64, 128, 128], overlap_size: List[int] = [16, 32, 32], pad_value:float=0):
        self.stride_size = [crop_size[0]-overlap_size[0], 
                            crop_size[1]-overlap_size[1], 
                            crop_size[2]-overlap_size[2]]
        self.overlap = overlap_size
        self.pad_value = pad_value

    def split(self, data):
        splits = []
        d, h, w = data.shape

        # Number of splits in each dimension
        nz = int(np.ceil(float(d) / self.stride_size[0]))
        ny = int(np.ceil(float(h) / self.stride_size[1]))
        nx = int(np.ceil(float(w) / self.stride_size[2]))

        nzyx = [nz, ny, nx]
        pad = [[0, int(nz * self.stride_size[0] + self.overlap[0] - d)],
                [0, int(ny * self.stride_size[1] + self.overlap[1] - h)],
                [0, int(nx * self.stride_size[2] + self.overlap[2] - w)]]

        data = np.pad(data, pad, 'constant', constant_values=self.pad_value)  

        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    start_z = int(iz * self.stride_size[0])
                    end_z = int((iz + 1) * self.stride_size[0] + self.overlap[0])
                    start_y = int(iy * self.stride_size[1])
                    end_y = int((iy + 1) * self.stride_size[1] + self.overlap[1])
                    start_x = int(ix * self.stride_size[2])
                    end_x = int((ix + 1) * self.stride_size[2] + self.overlap[2])

                    split = data[np.newaxis, np.newaxis, start_z:end_z, start_y:end_y, start_x:end_x]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits, nzyx

    def combine(self, output, nzhw):
        nz, nh, nw = nzhw
        idx = 0
        for iz in range(nz): 
            for ih in range(nh):
                for iw in range(nw):
                    sz = int(iz * self.stride_size[0])
                    sh = int(ih * self.stride_size[1])
                    sw = int(iw * self.stride_size[2])
                    # [N, 8]
                    # 8-> id, prob, z_min, y_min, x_min, d, h, w 
                    output[idx][:, 2] += sz
                    output[idx][:, 3] += sh
                    output[idx][:, 4] += sw
                    idx += 1
        return output
    
class AugSplitComb():
    """Augmented Split and Combine"""
    def __init__(self, 
                 crop_size: List[int] = [64, 128, 128],
                 overlap_size: List[int] = [16, 32, 32],
                 flip_axes: List[List[int]] = [[2]],
                 pad_value: float=0):
        self.stride_size = [crop_size[0] - overlap_size[0], 
                            crop_size[1] - overlap_size[1], 
                            crop_size[2] - overlap_size[2]]
        self.overlap = overlap_size
        self.pad_value = pad_value
        self.flip_axes = flip_axes
        
    def split(self, image):
        original_shape = image.shape
        d, h, w = image.shape

        # Number of splits in each dimension
        nz = int(np.ceil(float(d) / self.stride_size[0]))
        ny = int(np.ceil(float(h) / self.stride_size[1]))
        nx = int(np.ceil(float(w) / self.stride_size[2]))

        pad = [[0, int(nz * self.stride_size[0] + self.overlap[0] - d)],
                [0, int(ny * self.stride_size[1] + self.overlap[1] - h)],
                [0, int(nx * self.stride_size[2] + self.overlap[2] - w)]]

        img_flip_axes = copy.deepcopy(self.flip_axes)
        img_flip_axes = [[]] + img_flip_axes # add no flip case
        images = []
        for axes in img_flip_axes:
            if len(axes) > 0:
                padding_image = np.flip(image, axes)
            else:
                padding_image = image.copy()
            padding_image = np.pad(padding_image, pad, 'constant', constant_values=self.pad_value)
            images.append(padding_image)
        
        splits = []
        splits_flip_axes = []
        splits_start_zyx = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    start_z = int(iz * self.stride_size[0])
                    end_z = int((iz + 1) * self.stride_size[0] + self.overlap[0])
                    start_y = int(iy * self.stride_size[1])
                    end_y = int((iy + 1) * self.stride_size[1] + self.overlap[1])
                    start_x = int(ix * self.stride_size[2])
                    end_x = int((ix + 1) * self.stride_size[2] + self.overlap[2])

                    for image, flip_axes in zip(images, img_flip_axes):
                        split = image[np.newaxis, np.newaxis, start_z:end_z, start_y:end_y, start_x:end_x]
                        splits.append(split)
                        splits_flip_axes.append(flip_axes)
                        splits_start_zyx.append([start_z, start_y, start_x])

        splits = np.concatenate(splits, 0)
        
        return splits, splits_flip_axes, splits_start_zyx, original_shape

    def combine(self, output, splits_flip_axes, splits_start_zyx, image_shape: Tuple[int, int, int]):
        if not isinstance(image_shape, np.ndarray):
            image_shape = np.array(image_shape)
        
        for idx in range(len(splits_flip_axes)):
            sz, sy, sx = splits_start_zyx[idx]
            flip_axes = splits_flip_axes[idx]
            # [N, 8]
            # 8-> id, prob, z_min, y_min, x_min, d, h, w
            output[idx][:, 2] = output[idx][:, 2] + sz
            output[idx][:, 3] = output[idx][:, 3] + sy
            output[idx][:, 4] = output[idx][:, 4] + sx
            if len(flip_axes) > 0:
                flip_axes = np.array(flip_axes) + 2 # 2 is the start of zyx
                output[idx][:, 1] = output[idx][:, 1] * (0.9 ** len(flip_axes))
                output[idx][:, flip_axes] = image_shape[flip_axes - 2] - 1 - output[idx][:, flip_axes]
        return output