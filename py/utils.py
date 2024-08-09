import random
import math
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import numpy as np
import hashlib
import glob
from PIL import Image, ImageFilter, ImageChops, ImageDraw, ImageOps, ImageEnhance, ImageFont
from typing import Union, List

def log(message:str, message_type:str='info'):
    name = 'FoxTools'

    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"# FoxTools: {name} -> {message}")


def generate_random_name(prefix:str, suffix:str, length:int) -> str:
    name = ''.join(random.choice("abcdefghijklmnopqrstupvxyz1234567890") for x in range(length))
    return prefix + name + suffix

def watermark_image_size(image:Image.Image) -> int:
    size = int(math.sqrt(image.width * image.height * 0.015625) * 0.9)
    return size

def tensor_to_image(image):
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert('RGB'))

def image_to_tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    

# PIL Hex
def pil2hex(image):
    return hashlib.sha256(np.array(tensor2pil(image)).astype(np.uint16).tobytes()).hexdigest()

# PIL to Mask
def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask

# Mask to PIL
def mask2pil(mask):
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = mask.cpu().numpy().astype('uint8')
    mask_pil = Image.fromarray(mask_np, mode="L")
    return mask_pil

# Tensor to SAM-compatible NumPy
def tensor2sam(image):
    # Convert tensor to numpy array in HWC uint8 format with pixel values in [0, 255]
    sam_image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    # Transpose the image to HWC format if it's in CHW format
    if sam_image.shape[0] == 3:
        sam_image = np.transpose(sam_image, (1, 2, 0))
    return sam_image

# SAM-compatible NumPy to tensor
def sam2tensor(image):
    # Convert the image to float32 and normalize the pixel values to [0, 1]
    float_image = image.astype(np.float32) / 255.0
    # Transpose the image from HWC format to CHW format
    chw_image = np.transpose(float_image, (2, 0, 1))
    # Convert the numpy array to a tensor
    tensor_image = torch.from_numpy(chw_image)
    return tensor_image

def tensor2pilex(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pilex(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

# SHA-256 Hash
def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()