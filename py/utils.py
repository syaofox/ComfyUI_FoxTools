import random
import math
from PIL import Image, ImageFilter, ImageChops, ImageDraw, ImageOps, ImageEnhance, ImageFont


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
    print(f"# 😺dzNodes: {name} -> {message}")


def generate_random_name(prefix:str, suffix:str, length:int) -> str:
    name = ''.join(random.choice("abcdefghijklmnopqrstupvxyz1234567890") for x in range(length))
    return prefix + name + suffix

def watermark_image_size(image:Image.Image) -> int:
    size = int(math.sqrt(image.width * image.height * 0.015625) * 0.9)
    return size