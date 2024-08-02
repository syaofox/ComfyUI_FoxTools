
import torch
import numpy as np
import os
import folder_paths
from PIL import Image
import re
from .utils import log


class MakeBatchFromImageList:
# based on ImageListToImageBatch by DrLtData

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_list": ("IMAGE", ),}}

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image_batch",) 
    INPUT_IS_LIST = True
    FUNCTION = "make_batch"
    CATEGORY = 'FoxTools'
   
    def make_batch(self, image_list):    
        
    
        if len(image_list) <= 1:
            return (image_list,)
            
        batched_images = torch.cat(image_list, dim=0)    

        return (batched_images, )     
    
NODE_CLASS_MAPPINGS = {
    "Foxtools: BatchImageFromList": MakeBatchFromImageList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Foxtools: BatchImageFromList": "Foxtools: BatchImageFromList"
}