import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter,ImageChops
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
    

class CreateBlurBord:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {               
                "width": ("INT",  { "default": 1024, "min": 0, "max": 14096, "step": 1 }),
                "height": ("INT",  { "default": 1024, "min": 0, "max": 14096, "step": 1 }),
                "board_percent": ("FLOAT",  { "default":0.1, "min": 0, "max":14096, "step": 0.01 }),
                "blur_radius": ("INT",  { "default": 20.0, "min": 0, "max": 14096, "step": 1 }),
            }, 
        }

    RETURN_TYPES = ("IMAGE","MASK", )
    FUNCTION = "blurbord"
    CATEGORY = "FoxTools"

    def blurbord(self, width, height,board_percent, blur_radius):
        print("blurbord",width,height)

        # 创建白色图片
        image = Image.new("RGB", (width, height), "white")        
        # 计算边框粗细
        border_thickness = int(min(width, height) * board_percent)        
        # 添加黑色边框
        image_with_border = ImageOps.expand(image, border=border_thickness, fill="black")        
        # # 模糊边框
        # blurred_image = image_with_border.filter(ImageFilter.GaussianBlur(radius=border_thickness))

        blurred_image = image_with_border.filter(ImageFilter.GaussianBlur(radius=blur_radius)) 
        tensor_blurred = image_to_tensor(blurred_image)
        blurred_image = tensor_blurred.unsqueeze(0)

        mask = blurred_image[:, :, :, 0]
        return (blurred_image, mask, )
    
class TrimBlackBoard:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {               
                "image1": ("IMAGE", ),
                "threshold":  ("INT",  { "default": 10, "min": 0, "max": 14096, "step": 1 }),
            }, 
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "func"
    CATEGORY = "FoxTools"

    def func(self, image1, threshold):

        img = tensor_to_image(image1[0])
        img = Image.fromarray(img)

        width, height = img.size
        
        # 检测顶部黑边
        top_offset = 0
        for y in range(height):
            for x in range(width):
                if img.getpixel((x, y))[0] > threshold or img.getpixel((x, y))[1] > threshold or img.getpixel((x, y))[2] > threshold: # type: ignore
                    top_offset = y
                    break
            if top_offset != 0:
                break
        
        # 检测底部黑边
        bottom_offset = 0
        for y in range(height-1, -1, -1):
            for x in range(width):
                if img.getpixel((x, y))[0] > threshold or img.getpixel((x, y))[1] > threshold or img.getpixel((x, y))[2] > threshold: # type: ignore
                    bottom_offset = y
                    break
            if bottom_offset != 0:
                break
        
        # 检测左侧黑边
        left_offset = 0
        for x in range(width):
            for y in range(height):
                if img.getpixel((x, y))[0] > threshold or img.getpixel((x, y))[1] > threshold or img.getpixel((x, y))[2] > threshold: # type: ignore
                    left_offset = x
                    break
            if left_offset != 0:
                break
        
        # 检测右侧黑边
        right_offset = 0
        for x in range(width-1, -1, -1):
            for y in range(height):
                if img.getpixel((x, y))[0] > threshold or img.getpixel((x, y))[1] > threshold or img.getpixel((x, y))[2] > threshold: # type: ignore
                    right_offset = x
                    break
            if right_offset != 0:
                break
        
        # 裁剪图像，去除黑边
        img = img.crop((left_offset, top_offset, right_offset + 1, bottom_offset + 1))
        croped_image = image_to_tensor(img).unsqueeze(0)
        return (croped_image, )
    

class ImageRotate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_from": ("IMAGE", ),
                "angle": ("FLOAT", { "default":0.1, "min": -14096, "max":14096, "step": 0.01 }),
                "expand": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rotated_image",)
    FUNCTION = "face_rotate"
    CATEGORY = "FoxTools"

    def face_rotate(self, image_from, angle,expand):
        image_from = tensor_to_image(image_from[0])
       
        image_from = Image.fromarray(image_from).rotate(angle,expand=expand)
        image_from = image_to_tensor(image_from).unsqueeze(0)

        return (image_from,)

    

NODE_CLASS_MAPPINGS = {
    "Foxtools: ImageAdd": ImageAdd,
    "Foxtools: CreateBlurBord": CreateBlurBord,
    "Foxtools: TrimBlackBoard": TrimBlackBoard,
    "Foxtools: ImageRotate": ImageRotate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Foxtools: ImageAdd": "Foxtools: ImageAdd",
    "Foxtools: CreateBlurBord": "Foxtools: CreateBlurBord",
    "Foxtools: TrimBlackBoard": "Foxtools: TrimBlackBoard",
    "Foxtools: ImageRotate": "Foxtools: ImageRotate",
}