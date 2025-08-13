import cv2
import math
import torch
import scipy
import imageio
import numpy as np

from monai import transforms as monai_transforms
from typing import Callable


def resize_volume(vol, size, max_frames, nearest_neighbor=False):
    W, H, F = vol.shape
    zoom_rate = size / W
    vol_reshape = scipy.ndimage.zoom(
        vol, (zoom_rate, zoom_rate, zoom_rate), order=3 if not nearest_neighbor else 0
    )
    resizeW, resizeH, resizeF = vol_reshape.shape
    if resizeF > max_frames:
        vol_reshape = vol_reshape[:, :, :max_frames]
        resizeF = max_frames
    else:
        resized_max_fr = int(math.ceil(max_frames * zoom_rate))
        vol_reshape = np.concatenate([vol_reshape, np.zeros((resizeW, resizeH, resized_max_fr - resizeF))], axis=-1)
    return vol_reshape, resizeF, zoom_rate


def process_volume(vol: np.ndarray, keep_frames: Callable = lambda x: x > 0.025):
    initial_resize = monai_transforms.ResizeWithPadOrCrop((512, 512))
    transform = monai_transforms.CropForeground(keys=["pixel_values"], source_key="pixel_values", return_coords=True)
    crop_vol, start_coords, end_coords = transform(vol)
    keep_frames_indices = np.where(keep_frames(np.mean(np.mean(crop_vol, axis=-1), axis=-1)))[0]
    crop_vol = crop_vol[keep_frames_indices]
    W, H, F = crop_vol.shape
    proc_vol = cv2.equalizeHist(crop_vol.reshape(W, -1).astype(np.uint8)).reshape(W, H, F)
    proc_vol = initial_resize(proc_vol).detach().cpu().numpy().transpose((1, 2, 0))
    proc_vol, max_fr = resize_volume(proc_vol, 256, max_frames=512)[:2]

    images = []
    val_transform = monai_transforms.Compose([monai_transforms.Resized(keys=['image'], spatial_size=(256, 256), mode=['bilinear'])])
    for i in range(proc_vol.shape[2]):
        image = torch.from_numpy(proc_vol[:, :, i]).unsqueeze(0)
        image_transformed = val_transform({"image": image})["image"]
        images.append(image_transformed)
    images = torch.stack(images)
    if images.max() > 1:
        images = images / 255.0
    # make the images three channels
    images = images.repeat(1, 3, 1, 1)
    return images, max_fr, keep_frames_indices


def load_tif_images(file_path):
    vol = imageio.imread(file_path)
    if np.max(vol) <= 1:
        vol = vol * 255
    return vol


def process_ct(ct_path: str):
    vol = load_tif_images(ct_path)
    images, frame_indices, keep_frames_indices = process_volume(vol, keep_frames=lambda x: x > 0.025)
    return images, frame_indices, keep_frames_indices