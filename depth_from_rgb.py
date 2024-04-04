import argparse
import cv2
import time
import math
import numpy
import torch
import torchvision.transforms
from torch import Tensor

from depth_anything import DepthAnything


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

    rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)
    rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32)
    rgb_resize = torchvision.transforms.Resize((depth_height, depth_width))

    i, t0 = 0, time.time()
    while video_rgb.isOpened():
        i += 1
        if i % 10 == 0:
            print(f'Frame {i} of {frame_count}, FPS: {i / (time.time() - t0):.2f}')

        is_frame, frame_rgb = video_rgb.read()
        if not is_frame: break

        frame_rgb = torch.tensor(frame_rgb, device=device, dtype=torch.float32)
        frame_rgb = frame_rgb.permute(2, 0, 1)
        frame_rgb = rgb_resize.forward(frame_rgb)
        frame_rgb = frame_rgb[[2, 1, 0]]
        frame_rgb /= 255
        frame_rgb -= rgb_mean[:, None, None]
        frame_rgb /= rgb_std[:, None, None]
        frame_rgb = frame_rgb.unsqueeze(0)
        frame_depth = depth_model.forward(frame_rgb)
        frame_depth -= frame_depth.min()
        frame_depth /= frame_depth.max()
        frame_depth = (frame_depth[0] * 255).to(torch.uint8)

        frame_depth = frame_depth.cpu().numpy()
        frame_depth = cv2.cvtColor(frame_depth, cv2.COLOR_GRAY2BGR)
        video_depth.write(frame_depth)

    video_rgb.release()
    video_depth.release()
