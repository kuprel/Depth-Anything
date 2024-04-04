import argparse
import cv2
import time
import math
import torch
import torchvision.transforms

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
    batch_size = 8

    path_rgb = args.video_path
    path_depth = path_rgb.replace('.mp4', '_depth.mp4')

    video_rgb = cv2.VideoCapture(path_rgb)
    frame_width, frame_height = int(video_rgb.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    depth_width = frame_width / frame_height * depth_height
    depth_width = math.ceil(depth_width / 14) * 14
    print("Depth Shape", (depth_height, depth_width))
    # frame_rate = int(video_rgb.get(cv2.CAP_PROP_FPS))
    frame_rate = 30
    frame_count = int(video_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = frame_count // batch_size * batch_size
    video_depth = cv2.VideoWriter(path_depth, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (depth_width, depth_height))

    msg = "Frame height: {}, Frame width: {}, Frame rate: {}, Frame count: {}"
    print(msg.format(frame_height, frame_width, frame_rate, frame_count))

    rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)
    rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32)
    rgb_resize = torchvision.transforms.Resize((depth_height, depth_width))

    t0 = time.time()
    for i in range(frame_count):
        if i % batch_size == 0:
            print(f'Frame {i} of {frame_count}, FPS: {i / (time.time() - t0):.2f}')

        frames_rgb = []
        for i in range(batch_size):
            is_frame, frame = video_rgb.read()
            if not is_frame: break
            frames_rgb.append(frame)

        frames_rgb = torch.stack([
            torch.tensor(frame, device=device, dtype=torch.float32)
            for frame in frames_rgb
        ])
        if i == 0: print('frames shape', frames_rgb.shape)
        frames_rgb = frames_rgb.permute(0, 3, 1, 2).flip(1)
        if i == 0: print('frames shape', frames_rgb.shape)

        # frame_rgb = frame_rgb.permute(2, 0, 1).flip(0)
        print(frames_rgb.shape)
        frames_rgb = rgb_resize.forward(frames_rgb)
        print(frames_rgb.shape)
        frames_rgb /= 255
        frames_rgb -= rgb_mean[None, :, None, None]
        frames_rgb /= rgb_std[None, :, None, None]

        with torch.no_grad():
            frames_depth = depth_model.forward(frames_rgb)

        frames_depth -= frames_depth.min()
        frames_depth /= frames_depth.max()
        frames_depth *= 255
        frames_depth = frames_depth.to(torch.uint8)

        frames_depth = frames_depth.cpu().numpy()
        for frame_depth in frames_depth:
            frame_depth = cv2.cvtColor(frame_depth, cv2.COLOR_GRAY2BGR)
            video_depth.write(frame_depth)

    video_rgb.release()
    video_depth.release()
