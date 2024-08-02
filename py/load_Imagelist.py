import torch
import numpy as np
import os
import re
from PIL import Image
from .utils import log  # 确保这个 utils 模块在你的代码路径中

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0) 

NODE_NAME = 'LoadImageList'

class LoadImageList:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "input_path": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "make_list"
    CATEGORY = "FoxTools"

    def make_list(self, start_index, max_images, input_path):
        # 检查输入路径是否存在
        if not os.path.exists(input_path):
            log(f"Image List: The input_path `{input_path}` does not exist", message_type="warning")
            return ("",)

        in_path = input_path

        # 检查文件夹是否为空
        if not os.listdir(in_path):
            log(f"Image List: The folder `{in_path}` is empty", message_type="warning")
            return None

        # 对文件列表进行排序
        file_list = sorted(os.listdir(in_path), key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()))

        image_list = []
        
        # 确保 start_index 在列表范围内
        start_index = max(0, min(start_index, len(file_list) - 1))

        # 计算结束索引
        end_index = min(start_index + max_images, len(file_list))

        for num in range(start_index, end_index):
            fname = os.path.join(in_path, file_list[num])
            img = Image.open(fname)
            image = img.convert("RGB")
            image_list.append(pil2tensor(image))

        if not image_list:
            log("Load Image List: No images found.", message_type="warning")
            return None

        # 将图像列表合并为一个张量
        images = torch.cat(image_list, dim=0)
        images_out = [images[i:i + 1, ...] for i in range(images.shape[0])]

        return (images_out, )

NODE_CLASS_MAPPINGS = {
    "Foxtools: LoadImageList": LoadImageList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Foxtools: LoadImageList": "Foxtools: LoadImageList"
}