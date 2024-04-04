import argparse
import cv2
import time
import math
import torch
from torch import Tensor
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_model = 'LiheYoung/depth_anything_{}14'.format(args.encoder)
    depth_model = DepthAnything.from_pretrained(depth_model)
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
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
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
    video_depth = cv2.VideoWriter(path_depth, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (depth_height, depth_width))

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
