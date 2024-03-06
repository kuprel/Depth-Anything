import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from PIL import Image

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def generate_stereo(left_img, depth, ipd):
    monitor_w = 38.5
    h, w, _ = left_img.shape
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    right = np.zeros_like(left_img)
    deviation_cm = ipd * 0.12
    deviation = deviation_cm * monitor_w * (w / 1920)
    col_r_shift = (1 - depth_normalized ** 2) * deviation
    col_r_indices = np.arange(w) - col_r_shift.astype(int)
    valid_indices = col_r_indices >= 0
    for row in range(h):
        valid_cols = col_r_indices[row, valid_indices[row]]
        right[row, valid_cols] = left_img[row, np.arange(w)[valid_indices[row]]]

    right_fix = right.copy()
    gray = cv2.cvtColor(right_fix, cv2.COLOR_BGR2GRAY)
    missing_pixels = np.where(gray == 0)
    for row, col in zip(*missing_pixels):
        for offset in range(1, int(deviation)):
            r_offset = min(col + offset, w - 1)
            l_offset = max(col - offset, 0)
            if not np.all(right_fix[row, r_offset] == 0):
                right_fix[row, col] = right_fix[row, r_offset]
                break
            elif not np.all(right_fix[row, l_offset] == 0):
                right_fix[row, col] = right_fix[row, l_offset]
                break

    return right_fix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = os.listdir(args.video_path)
        filenames = [os.path.join(args.video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()

    for k, filename in enumerate(filenames):
        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)

        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2

        filename = os.path.basename(filename)
        output_path = '../stereo.mp4'
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))

        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            frame_pil =  Image.fromarray((frame * 255).astype(np.uint8))

            frame = transform({'image': frame})['image']
            frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                depth = depth_anything(frame)

            depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            depth_map = depth.cpu().numpy().astype(np.uint8)
            left_img = np.array(frame_pil)
            depth_map = cv2.blur(depth_map, (3, 3))
            ipd = 6.34
            right_img = generate_stereo(left_img, depth_map, ipd)
            stereo = np.hstack([left_img, right_img])
            stereo_bgr = cv2.cvtColor(stereo, cv2.COLOR_RGB2BGR)

            depth = depth.cpu().numpy().astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

            out.write(stereo_bgr)

        raw_video.release()
        out.release()
