import glob
from random import random
import torch
import os
import re
import numpy as np
from PIL import Image, ImageOps, ImageFilter,ImageChops
from PIL.PngImagePlugin import PngInfo
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from .utils import log, generate_random_name, pil2tensor,tensor_to_image,image_to_tensor, get_sha256,tensor2pil
import datetime
import json
import folder_paths
import shutil
from comfy.utils import  common_upscale


NODE_FILE = os.path.abspath(__file__)
WAS_SUITE_ROOT = os.path.dirname(NODE_FILE)
WAS_DATABASE = os.path.join(WAS_SUITE_ROOT, 'was_suite_settings.json')
WAS_HISTORY_DATABASE = os.path.join(WAS_SUITE_ROOT, 'was_history.json')

ALLOWED_EXT = ('.jpeg', '.jpg', '.png','.tiff', '.gif', '.bmp', '.webp')


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


def update_history_images(new_paths):
    HDB = WASDatabase(WAS_HISTORY_DATABASE)
    if HDB.catExists("History") and HDB.keyExists("History", "Images"):
        saved_paths = HDB.get("History", "Images")
        for path_ in saved_paths:
            if not os.path.exists(path_):
                saved_paths.remove(path_)
        if isinstance(new_paths, str):
            if new_paths in saved_paths:
                saved_paths.remove(new_paths)
            saved_paths.append(new_paths)
        elif isinstance(new_paths, list):
            for path_ in new_paths:
                if path_ in saved_paths:
                    saved_paths.remove(path_)
                saved_paths.append(path_)
        HDB.update("History", "Images", saved_paths)
    else:
        if not HDB.catExists("History"):
            HDB.insertCat("History")
        if isinstance(new_paths, str):
            HDB.insert("History", "Images", [new_paths])
        elif isinstance(new_paths, list):
            HDB.insert("History", "Images", new_paths)

# WAS SETTINGS MANAGER

class WASDatabase:
    """
    The WAS Suite Database Class provides a simple key-value database that stores
    data in a flatfile using the JSON format. Each key-value pair is associated with
    a category.

    Attributes:
        filepath (str): The path to the JSON file where the data is stored.
        data (dict): The dictionary that holds the data read from the JSON file.

    Methods:
        insert(category, key, value): Inserts a key-value pair into the database
            under the specified category.
        get(category, key): Retrieves the value associated with the specified
            key and category from the database.
        update(category, key): Update a value associated with the specified
            key and category from the database.
        delete(category, key): Deletes the key-value pair associated with the
            specified key and category from the database.
        _save(): Saves the current state of the database to the JSON file.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        try:
            with open(filepath, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}

    def catExists(self, category):
        return category in self.data

    def keyExists(self, category, key):
        return category in self.data and key in self.data[category]

    def insert(self, category, key, value):
        if not isinstance(category, str) or not isinstance(key, str):
            log("Category and key must be strings", message_type="warning")
            return

        if category not in self.data:
            self.data[category] = {}
        self.data[category][key] = value
        self._save()

    def update(self, category, key, value):
        if category in self.data and key in self.data[category]:
            self.data[category][key] = value
            self._save()

    def updateCat(self, category, dictionary):
        self.data[category].update(dictionary)
        self._save()

    def get(self, category, key):
        return self.data.get(category, {}).get(key, None)

    def getDB(self):
        return self.data

    def insertCat(self, category):
        if not isinstance(category, str):
            log("Category must be a string", message_type="warning")
            return

        if category in self.data:
            log(f"The database category '{category}' already exists!", message_type="warning")
            return
        self.data[category] = {}
        self._save()

    def getDict(self, category):
        if category not in self.data:
            log(f"The database category '{category}' does not exist!", message_type="warning")
            return {}
        return self.data[category]

    def delete(self, category, key):
        if category in self.data and key in self.data[category]:
            del self.data[category][key]
            self._save()

    def _save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=4)
        except FileNotFoundError:
            log(f"Cannot save database to file '{self.filepath}'. "
                 "Storing the data in the object instead. Does the folder and node file have write permissions?", message_type="warning")
        except Exception as e:
            log(f"Error while saving JSON data: {e}", message_type="warning")

# Initialize the settings database
WDB = WASDatabase(WAS_DATABASE)

class LoadImageBatch:
    def __init__(self):
        self.HDB = WASDatabase(WAS_HISTORY_DATABASE)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_image", "incremental_image", "random"],),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "label": ("STRING", {"default": 'Batch 001', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "pattern": ("STRING", {"default": '*', "multiline": False}),
                "allow_RGBA_output": (["false","true"],),
            },
            "optional": {
                "filename_text_extension": (["true", "false"],),
            }
        }

    RETURN_TYPES = ("IMAGE","STRING")
    RETURN_NAMES = ("image","filename_text")
    FUNCTION = "load_batch_images"

    CATEGORY = "FoxTools/Images"

    def load_batch_images(self, path, pattern='*', index=0, mode="single_image", label='Batch 001', allow_RGBA_output='false', filename_text_extension='true'):

        allow_RGBA_output = (allow_RGBA_output == 'true')

        if not os.path.exists(path):
            return (None, )
        fl = self.BatchImageLoader(path, label, pattern)
        new_paths = fl.image_paths
        if mode == 'single_image':
            image, filename = fl.get_image_by_id(index) # type: ignore
            if image == None:
                log(f"No valid image was found for the inded `{index}`", message_type="warning")
                return (None, None)
        elif mode == 'incremental_image':
            image, filename = fl.get_next_image()
            if image == None:
                log(f"No valid image was found for the next ID. Did you remove images from the source directory?", message_type="warning")
                return (None, None)
        else:
            newindex = int(random.random() * len(fl.image_paths))
            image, filename = fl.get_image_by_id(newindex) # type: ignore
            if image == None:
                log(f"No valid image was found for the next ID. Did you remove images from the source directory?", message_type="warning")
                return (None, None)


        # Update history
        update_history_images(new_paths)

        if not allow_RGBA_output:
           image = image.convert("RGB")

        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]

        return (pil2tensor(image), filename)

    class BatchImageLoader:
        def __init__(self, directory_path, label, pattern):
            self.WDB = WDB
            self.image_paths = []
            self.load_images(directory_path, pattern)
            self.image_paths.sort()
            stored_directory_path = self.WDB.get('Batch Paths', label)
            stored_pattern = self.WDB.get('Batch Patterns', label)
            if stored_directory_path != directory_path or stored_pattern != pattern:
                self.index = 0
                self.WDB.insert('Batch Counters', label, 0)
                self.WDB.insert('Batch Paths', label, directory_path)
                self.WDB.insert('Batch Patterns', label, pattern)
            else:
                self.index = self.WDB.get('Batch Counters', label)
            self.label = label

        def load_images(self, directory_path, pattern):
            for file_name in glob.glob(os.path.join(glob.escape(directory_path), pattern), recursive=True):
                if file_name.lower().endswith(ALLOWED_EXT):
                    abs_file_path = os.path.abspath(file_name)
                    self.image_paths.append(abs_file_path)

        def get_image_by_id(self, image_id):
            if image_id < 0 or image_id >= len(self.image_paths):
                log(f"Invalid image index `{image_id}`", message_type="warning")
                return
            i = Image.open(self.image_paths[image_id])
            i = ImageOps.exif_transpose(i)
            return (i, os.path.basename(self.image_paths[image_id]))

        def get_next_image(self):
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            self.index += 1
            if self.index == len(self.image_paths):
                self.index = 0
            log(f'{self.label} Index: {self.index}', message_type="warning")
            self.WDB.insert('Batch Counters', self.label, self.index)
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            return (i, os.path.basename(image_path))

        def get_current_image(self):
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            return os.path.basename(image_path)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs['mode'] != 'single_image':
            return float("NaN")
        else:
            fl = LoadImageBatch.BatchImageLoader(kwargs['path'], kwargs['label'], kwargs['pattern'])
            filename = fl.get_current_image()
            image = os.path.join(kwargs['path'], filename)
            sha = get_sha256(image)
            return sha


class ImageExtractFromBatch:
    INPUT_TYPES = lambda: {
        "required": {
            "images": ("IMAGE",),
            "index": ("INT", {"default": 0, "min": 0, "step": 1}),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "FoxTools/Images"

    def execute(self, images: torch.Tensor, index: int):
        assert isinstance(images, torch.Tensor)
        assert isinstance(index, int)

        img = images[index].unsqueeze(0)

        return (img,)


class ImageTileBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_tiles": ("INT", {"default":4, "max": 64, "min":2, "step":1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGES",)
    FUNCTION = "tile_image"

    CATEGORY = "FoxTools/Images"

    def tile_image(self, image, num_tiles=6):
        image = tensor2pil(image.squeeze(0))
        img_width, img_height = image.size

        num_rows = int(num_tiles ** 0.5)
        num_cols = (num_tiles + num_rows - 1) // num_rows
        tile_width = img_width // num_cols
        tile_height = img_height // num_rows

        tiles = []
        for y in range(0, img_height, tile_height):
            for x in range(0, img_width, tile_width):
                tile = image.crop((x, y, x + tile_width, y + tile_height))
                tiles.append(pil2tensor(tile))

        tiles = torch.stack(tiles, dim=0).squeeze(1)

        return (tiles, )
    
class ImageConcanate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "direction": (
            [   'right',
                'down',
                'left',
                'up',
            ],
            {
            "default": 'right'
             }),
            "match_image_size": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concanate"
    CATEGORY = "FoxTools/Images"
    DESCRIPTION = """
Concatenates the image2 to image1 in the specified direction.
"""

    def concanate(self, image1, image2, direction, match_image_size, first_image_shape=None):
        # Check if the batch sizes are different
        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]

        if batch_size1 != batch_size2:
            # Calculate the number of repetitions needed
            max_batch_size = max(batch_size1, batch_size2)
            repeats1 = max_batch_size // batch_size1
            repeats2 = max_batch_size // batch_size2
            
            # Repeat the images to match the largest batch size
            image1 = image1.repeat(repeats1, 1, 1, 1)
            image2 = image2.repeat(repeats2, 1, 1, 1)

        if match_image_size:
            # Use first_image_shape if provided; otherwise, default to image1's shape
            target_shape = first_image_shape if first_image_shape is not None else image1.shape

            original_height = image2.shape[1]
            original_width = image2.shape[2]
            original_aspect_ratio = original_width / original_height

            if direction in ['left', 'right']:
                # Match the height and adjust the width to preserve aspect ratio
                target_height = target_shape[1]  # B, H, W, C format
                target_width = int(target_height * original_aspect_ratio)
            elif direction in ['up', 'down']:
                # Match the width and adjust the height to preserve aspect ratio
                target_width = target_shape[2]  # B, H, W, C format
                target_height = int(target_width / original_aspect_ratio)
            
            # Adjust image2 to the expected format for common_upscale
            image2_for_upscale = image2.movedim(-1, 1)  # Move C to the second position (B, C, H, W)
            
            # Resize image2 to match the target size while preserving aspect ratio
            image2_resized = common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")
            
            # Adjust image2 back to the original format (B, H, W, C) after resizing
            image2_resized = image2_resized.movedim(1, -1)
        else:
            image2_resized = image2

        # Concatenate based on the specified direction
        if direction == 'right':
            concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
        elif direction == 'down':
            concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
        elif direction == 'left':
            concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width
        elif direction == 'up':
            concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height
        return concatenated_image,





class ImageResizeByShorterSide:
    CATEGORY = "FoxTools/Images"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "size": ("INT", {"default": 512, "min": 0, "step": 1, "max": 99999}),
            "interpolation_mode": (
                ["bicubic", "bilinear", "nearest", "nearest exact"],
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        size: int,
        interpolation_mode: str,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(size, int)
        assert isinstance(interpolation_mode, str)

        interpolation_mode = interpolation_mode.upper().replace(" ", "_")
        interpolation_mode = getattr(InterpolationMode, interpolation_mode)

        image = image.permute(0, 3, 1, 2)
        image = F.resize(
            image,
            size, # type: ignore
            interpolation=interpolation_mode, # type: ignore
            antialias=True,
        )
        image = image.permute(0, 2, 3, 1)

        return (image,)



class ImageResizeByLongerSide:
    CATEGORY = "FoxTools/Images"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "size": ("INT", {"default": 512, "min": 0, "step": 1, "max": 99999}),
            "interpolation_mode": (
                ["bicubic", "bilinear", "nearest", "nearest exact"],
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        size: int,
        interpolation_mode: str,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(size, int)
        assert isinstance(interpolation_mode, str)

        interpolation_mode = interpolation_mode.upper().replace(" ", "_")
        interpolation_mode = getattr(InterpolationMode, interpolation_mode)

        _, h, w, _ = image.shape

        if h >= w:
            new_h = size
            new_w = round(w * new_h / h)
        else:  # h < w
            new_w = size
            new_h = round(h * new_w / w)

        image = image.permute(0, 3, 1, 2)
        image = F.resize(
            image,
            (new_h, new_w), # type: ignore
            interpolation=interpolation_mode, # type: ignore
            antialias=True,
        )
        image = image.permute(0, 2, 3, 1)

        return (image,)
    

NODE_CLASS_MAPPINGS = {
    "FoxBatchImageFromList": MakeBatchFromImageList,
    "FoxImageAdd": ImageAdd,
    "FoxCreateBlurBord": CreateBlurBord,
    "FoxTrimBlackBoard": TrimBlackBoard,
    "FoxImageRotate": ImageRotate,
    "FoxLoadImageList": LoadImageList,
    "FoxSaveImagePlus": SaveImagePlus,
    "FoxColorMatch": ColorMatch,
    "FoxLoadImageBatch": LoadImageBatch,
    "FoxImageExtractFromBatch": ImageExtractFromBatch,
    "FoxImageTileBatch": ImageTileBatch,
    "FoxImageConcanate": ImageConcanate,
    "FoxImageResizeByShorterSide": ImageResizeByShorterSide,
    "FoxImageResizeByLongerSide": ImageResizeByLongerSide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoxBatchImageFromList": "FoxTools: Batch Image From List",
    "FoxImageAdd": "FoxTools: Image Add",
    "FoxCreateBlurBord": "FoxTools: Create BlurBord",
    "FoxTrimBlackBoard": "FoxTools: Trim BlackBoard",
    "FoxImageRotate": "FoxTools: Image Rotate",
    "FoxLoadImageList": "FoxTools: Load ImageList",
    "FoxSaveImagePlus": "FoxTools: Save ImagePlus",
    "FoxColorMatch": "FoxTools: ColorMatch",
    "FoxLoadImageBatch": "FoxTools: Load Image Batch",
    "FoxImageExtractFromBatch": "FoxTools: Image Extract From Batch",
    "FoxImageTileBatch": "FoxTools: Image Tile Batch",
    "FoxImageConcanate": "FoxTools: Image Concanate",
    "FoxImageResizeByShorterSide": "FoxTools: Image Resize By Shorter Side",
    "FoxImageResizeByLongerSide": "FoxTools: Image Resize By Longer Side",
}

