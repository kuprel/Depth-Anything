import argparse
import cv2
import time
import math
import torch
from torch import Tensor
from torchvision.transforms import Compose

from depth_anything import DepthAnything

import torch.nn.functional as F

import numpy as np
import cv2
import math


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.

    Args:
        sample (dict): sample
        size (tuple): image size

    Returns:
        tuple: new size
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                # sample["semseg_mask"] = cv2.resize(
                #     sample["semseg_mask"], (width, height), interpolation=cv2.INTER_NEAREST
                # )
                sample["semseg_mask"] = F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...], (height, width), mode='nearest').numpy()[0, 0]

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                # sample["mask"] = sample["mask"].astype(bool)

        # print(sample['image'].shape, sample['depth'].shape)
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_model = DepthAnything(encoder=args.encoder)
    weights = torch.load('weights/depth_anything_{}14.pt'.format(args.encoder))
    depth_model.load_state_dict(weights)
    depth_model = depth_model.to(device).eval()

    depth_height = 518

    transform = Compose([
        Resize(
            width=depth_height,
            height=depth_height,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        )
    ])

    path_rgb = args.video_path
    path_depth = path_rgb.replace('.mp4', '_depth.mp4')

    video_rgb = cv2.VideoCapture(path_rgb)
    frame_width, frame_height = int(video_rgb.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    depth_width = frame_width / frame_height * depth_height
    depth_width = math.ceil(depth_width / 14) * 14
    print("Depth Width", depth_width)
    # frame_rate = int(video_rgb.get(cv2.CAP_PROP_FPS))
    frame_rate = 30
    frame_count = int(video_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    video_depth = cv2.VideoWriter(path_depth, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (depth_width, depth_height))

    msg = "Frame height: {}, Frame width: {}, Frame rate: {}, Frame count: {}"
    print(msg.format(frame_height, frame_width, frame_rate, frame_count))

    i, t0 = 0, time.time()
    while video_rgb.isOpened():
        i += 1
        if i % 10 == 0:
            print(f'Frame {i} of {frame_count}, FPS: {i / (time.time() - t0):.2f}')

        is_frame, frame_rgb = video_rgb.read()
        if not is_frame: break

        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB) / 255.0
        frame_rgb = transform({'image': frame_rgb})['image']
        frame_rgb -= np.array([0.485, 0.456, 0.406])
        frame_rgb /= np.array([0.229, 0.224, 0.225])
        frame_rgb = np.transpose(frame_rgb, (2, 0, 1))
        frame_rgb = np.ascontiguousarray(frame_rgb).astype(np.float32)
        frame_rgb = torch.from_numpy(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            frame_depth: Tensor = depth_model(frame_rgb)

        assert(depth_width == frame_depth.shape[-1])
        assert(depth_width == frame_rgb.shape[-1])

        frame_depth -= frame_depth.min()
        frame_depth /= frame_depth.max()
        frame_depth = (frame_depth[0] * 255).to(torch.uint8).cpu().numpy()
        frame_depth = cv2.cvtColor(frame_depth, cv2.COLOR_GRAY2BGR)

        video_depth.write(frame_depth)

    video_rgb.release()
    video_depth.release()
