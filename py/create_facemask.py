import cv2
import numpy as np
import os
import ctypes
import onnxruntime
from PIL import Image
from typing import Any, List, Optional
from functools import lru_cache
import torch
import torchvision.transforms.v2 as T

def tensor_to_image(image):
    if image.dim() == 4:
        image = image.squeeze(0)  # 移除批处理维度
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert('RGB'))

def image_to_tensor(image):

    return T.ToTensor()(image).permute(1, 2, 0)



def sanitize_path_for_windows(full_path: str) -> Optional[str]:
    buffer_size = 0

    while True:
        unicode_buffer = ctypes.create_unicode_buffer(buffer_size)
        buffer_threshold = ctypes.windll.kernel32.GetShortPathNameW(full_path, unicode_buffer, buffer_size)

        if buffer_size > buffer_threshold:
            return unicode_buffer.value
        if buffer_threshold == 0:
            return None
        buffer_size = buffer_threshold



def apply_execution_provider_options(execution_device_id: str, execution_providers: List[str]) -> List[Any]:
    execution_providers_with_options: List[Any] = []

    for execution_provider in execution_providers:
        if execution_provider == 'CUDAExecutionProvider':
            execution_providers_with_options.append((execution_provider, {
                'device_id': execution_device_id,
                'cudnn_conv_algo_search': 'EXHAUSTIVE'
            }))
        elif execution_provider == 'OpenVINOExecutionProvider':
            execution_providers_with_options.append((execution_provider, {
                'device_id': execution_device_id,
                'device_type': execution_device_id + '_FP32'
            }))
        elif execution_provider in ['DmlExecutionProvider', 'ROCMExecutionProvider']:
            execution_providers_with_options.append((execution_provider, {
                'device_id': execution_device_id
            }))
        else:
            execution_providers_with_options.append(execution_provider)
    return execution_providers_with_options

@lru_cache(maxsize=None)
def get_face_occluder():
    occluder_model_path = r"D:\AI\facestudio\res\models\face_occluder.onnx"
    return onnxruntime.InferenceSession(occluder_model_path, providers=apply_execution_provider_options('0', ["CUDAExecutionProvider"]))

def create_occlusion_mask(crop_vision_frame ):
    face_occluder = get_face_occluder()
    prepare_vision_frame = cv2.resize(crop_vision_frame, face_occluder.get_inputs()[0].shape[1:3][::-1])
    prepare_vision_frame = np.expand_dims(prepare_vision_frame, axis = 0).astype(np.float32) / 255
    prepare_vision_frame = prepare_vision_frame.transpose(0, 1, 2, 3)
    occlusion_mask  = face_occluder.run(None, {
    face_occluder.get_inputs()[0].name: prepare_vision_frame
    })[0][0]
    occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
    occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
    occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
    return occlusion_mask

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

NODE_NAME = 'CreateFaceMask'

class CreateFaceMask:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Mask",)
    FUNCTION = "generate_mask"
    CATEGORY = "FoxTools"

    def generate_mask(self, input_image):
        input_image = tensor_to_image(input_image[0])
        image_np = np.array(input_image)
        print(image_np.shape)
        occlusion_mask = create_occlusion_mask(image_np)

        image_to = torch.from_numpy(occlusion_mask).unsqueeze(0)

        # mask_image = Image.fromarray((occlusion_mask * 255).astype(np.uint8))

        # # image_to = image_to_tensor(mask_image).unsqueeze(0)
        # mask_image = Image.fromarray((image_np * 255).astype(np.uint8))
        # image_to = image_to_tensor(mask_image).unsqueeze(0)

        return (image_to, )

NODE_CLASS_MAPPINGS = {
    "FoxTools: CreateFaceMask": CreateFaceMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoxTools: CreateFaceMask": "FoxTools: CreateFaceMask"
}