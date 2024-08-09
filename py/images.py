import torch
import os
import re
import numpy as np
from PIL import Image, ImageOps, ImageFilter,ImageChops
from PIL.PngImagePlugin import PngInfo
import torchvision.transforms.v2 as T
from .utils import log, generate_random_name, pil2tensor,tensor_to_image,image_to_tensor
import datetime
import json
import folder_paths
import shutil


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
    FUNCTION = "run"
    CATEGORY = "FoxTools/Images"

    def run(self, image1,image2):

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
    FUNCTION = "run"
    CATEGORY = "FoxTools/Images"

    def run(self, width, height,board_percent, blur_radius):

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
    FUNCTION = "run"
    CATEGORY = "FoxTools/Images"

    def run(self, image1, threshold):

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
    FUNCTION = "run"
    CATEGORY = "FoxTools/Images"

    def run(self, image_from, angle,expand):
        image_from = tensor_to_image(image_from[0])
       
        image_from = Image.fromarray(image_from).rotate(angle,expand=expand)
        image_from = image_to_tensor(image_from).unsqueeze(0)

        return (image_from,)


class MakeBatchFromImageList:
# based on ImageListToImageBatch by DrLtData

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_list": ("IMAGE", ),}}

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image_batch",) 
    INPUT_IS_LIST = True
    FUNCTION = "run"
    CATEGORY = 'FoxTools/Images'
   
    def run(self, image_list):    
        
    
        if len(image_list) <= 1:
            return (image_list,)
            
        batched_images = torch.cat(image_list, dim=0)    

        return (batched_images, )     


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
    CATEGORY = "FoxTools/Images"

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
    
class SaveImagePlus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory() # type: ignore
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "custom_path": ("STRING", {"default": ""}),
                     "filename_prefix": ("STRING", {"default": "comfyui"}),
                     "timestamp": (["None", "second", "millisecond"],),
                     "format": (["png", "jpg"],),
                     "quality": ("INT", {"default": 80, "min": 10, "max": 100, "step": 1}),
                     "meta_data": ("BOOLEAN", {"default": False}),
                     "save_workflow_as_json": ("BOOLEAN", {"default": False}),
                     "preview": ("BOOLEAN", {"default": True}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_image_plus"
    OUTPUT_NODE = True
    CATEGORY = 'FoxTools/Images'

    def save_image_plus(self, images, custom_path, filename_prefix, timestamp, format, quality,
                           meta_data,  preview, save_workflow_as_json,
                           prompt=None, extra_pnginfo=None):

        now = datetime.datetime.now()
        custom_path = custom_path.replace("%date", now.strftime("%Y-%m-%d"))
        custom_path = custom_path.replace("%time", now.strftime("%H-%M-%S"))
        filename_prefix = filename_prefix.replace("%date", now.strftime("%Y-%m-%d"))
        filename_prefix = filename_prefix.replace("%time", now.strftime("%H-%M-%S"))
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]) # type: ignore
        results = list()
        temp_sub_dir = generate_random_name('_savepreview_', '_temp', 16)
        temp_dir = os.path.join(folder_paths.get_temp_directory(), temp_sub_dir) # type: ignore
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # if blind_watermark != "":
            #     img_mode = img.mode
            #     wm_size = watermark_image_size(img)
            #     import qrcode
            #     qr = qrcode.QRCode(
            #         version=1,
            #         error_correction=qrcode.constants.ERROR_CORRECT_H,
            #         box_size=20,
            #         border=1,
            #     )
            #     qr.add_data(blind_watermark.encode('utf-8'))
            #     qr.make(fit=True)
            #     qr_image = qr.make_image(fill_color="black", back_color="white")
            #     qr_image = qr_image.resize((wm_size, wm_size), Image.BICUBIC).convert("L")

            #     y, u, v, _ = image_channel_split(img, mode='YCbCr')
            #     _u = add_invisibal_watermark(u, qr_image)
            #     wm_img = image_channel_merge((y, _u, v), mode='YCbCr')

            #     if img.mode == "RGBA":
            #         img = RGB2RGBA(wm_img, img.split()[-1])
            #     else:
            #         img = wm_img.convert(img_mode)

            metadata = None
            if meta_data:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            if timestamp == "millisecond":
                file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]}'
            elif timestamp == "second":
                file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S")}'
            else:
                file = f'{filename}_{counter:05}'


            preview_filename = ""
            if custom_path != "":
                if not os.path.exists(custom_path):
                    try:
                        os.makedirs(custom_path)
                    except Exception as e:
                        log(f"skipped, because unable to create temporary folder.",
                            message_type='warning')
                        raise FileNotFoundError(f"cannot create custom_path {custom_path}, {e}")

                full_output_folder = os.path.normpath(custom_path)
                # save preview image to temp_dir
                if os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir)
                try:
                    os.makedirs(temp_dir)
                except Exception as e:
                    print(e)
                    log(f"skipped, because unable to create temporary folder.",
                        message_type='warning')
                try:
                    preview_filename = os.path.join(generate_random_name('saveimage_preview_', '_temp', 16) + '.png')
                    img.save(os.path.join(temp_dir, preview_filename))
                except Exception as e:
                    print(e)
                    log(f"skipped, because unable to create temporary file.", message_type='warning')

            # check if file exists, change filename
            while os.path.isfile(os.path.join(full_output_folder, f"{file}.{format}")):
                counter += 1
                if timestamp == "millisecond":
                    file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]}_{counter:05}'
                elif timestamp == "second":
                    file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S")}_{counter:05}'
                else:
                    file = f"{filename}_{counter:05}"

            image_file_name = os.path.join(full_output_folder, f"{file}.{format}")
            json_file_name = os.path.join(full_output_folder, f"{file}.json")

            if format == "png":
                img.save(image_file_name, pnginfo=metadata, compress_level= (100 - quality) // 10)
            else:
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(image_file_name, quality=quality)
            log(f"Saving image to {image_file_name}")

            if save_workflow_as_json:
                try:
                    workflow = (extra_pnginfo or {}).get('workflow')
                    if workflow is None:
                        log('No workflow found, skipping saving of JSON')
                    with open(f'{json_file_name}', 'w') as workflow_file:
                        json.dump(workflow, workflow_file)
                        log(f'Saved workflow to {json_file_name}')
                except Exception as e:
                    log(
                        f'Failed to save workflow as json due to: {e}, proceeding with the remainder of saving execution', message_type="warning")

            if preview:
                if custom_path == "":
                    results.append({
                        "filename": f"{file}.{format}",
                        "subfolder": subfolder,
                        "type": self.type
                    })
                else:
                    results.append({
                        "filename": preview_filename,
                        "subfolder": temp_sub_dir,
                        "type": "temp"
                    })

            counter += 1

        return { "ui": { "images": results } }


class ColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
            [   
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
            ], {
               "default": 'mkl'
            }),
                
            },
        }
    
    CATEGORY = 'FoxTools/Images'

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
    color-matcher enables color transfer across images which comes in handy for automatic  
    color-grading of photographs, paintings and film sequences as well as light-field  
    and stopmotion corrections.  

    The methods behind the mappings are based on the approach from Reinhard et al.,  
    the Monge-Kantorovich Linearization (MKL) as proposed by Pitie et al. and our analytical solution  
    to a Multi-Variate Gaussian Distribution (MVGD) transfer in conjunction with classical histogram   
    matching. As shown below our HM-MVGD-HM compound outperforms existing methods.   
    https://github.com/hahnec/color-matcher/

    """
    
    def colormatch(self, image_ref, image_target, method):
        try:
            from color_matcher import ColorMatcher
        except:
            raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
        cm = ColorMatcher()
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        out = []
        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

        for i in range(batch_size):
            image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
            except BaseException as e:
                print(f"Error occurred during transfer: {e}")
                break
            out.append(torch.from_numpy(image_result))
            
        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)



      
NODE_CLASS_MAPPINGS = {
    "Foxtools: BatchImageFromList": MakeBatchFromImageList,
    "Foxtools: ImageAdd": ImageAdd,
    "Foxtools: CreateBlurBord": CreateBlurBord,
    "Foxtools: TrimBlackBoard": TrimBlackBoard,
    "Foxtools: ImageRotate": ImageRotate,
    "Foxtools: LoadImageList": LoadImageList,
    "Foxtools: SaveImagePlus": SaveImagePlus,
    "Foxtools: ColorMatch": ColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Foxtools: BatchImageFromList": "Foxtools: BatchImageFromList",
    "Foxtools: ImageAdd": "Foxtools: ImageAdd",
    "Foxtools: CreateBlurBord": "Foxtools: CreateBlurBord",
    "Foxtools: TrimBlackBoard": "Foxtools: TrimBlackBoard",
    "Foxtools: ImageRotate": "Foxtools: ImageRotate",
    "Foxtools: LoadImageList": "Foxtools: LoadImageList",
    "Foxtools: SaveImagePlus": "Foxtools: SaveImagePlus",
    "Foxtools: ColorMatch": "Foxtools: ColorMatch"
}

