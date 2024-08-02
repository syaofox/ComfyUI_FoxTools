import torch
import numpy as np
import os
import re
from PIL import Image
from .utils import log  # 确保这个 utils 模块在你的代码路径中
import torchvision.transforms.v2 as T

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0) 

NODE_NAME = 'ImageAdd'

def tensor_to_image(image):
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert('RGB'))

def image_to_tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)
    #return T.ToTensor()(Image.fromarray(image)).permute(1, 2, 0)


class ImageAdd:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),

            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "op_image_add"
    CATEGORY = "FoxTools"

    def op_image_add(self, image1,image2):

        # Convert tensors to images
        overlay_array = tensor_to_image(image1[0])
        base_array = tensor_to_image(image2[0])

        # Create mask where overlay is not black
        mask = (overlay_array[:, :, 0:3] != [0, 0, 0]).any(axis=2)

        # Overlay the images using the mask
        base_array[mask] = overlay_array[mask]

        # Convert the result back to a tensor
        result_image = image_to_tensor(Image.fromarray(base_array)).unsqueeze(0)

        return (result_image, )

NODE_CLASS_MAPPINGS = {
    "Foxtools: ImageAdd": ImageAdd
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Foxtools: ImageAdd": "Foxtools: ImageAdd"
}