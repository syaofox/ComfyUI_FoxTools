import cv2
import numpy as np
import onnxruntime
from typing import Any, List
import torch
import torchvision.transforms.v2 as T

def tensor_to_image(image):
    if image.dim() == 4:
        image = image.squeeze(0)  # 移除批处理维度
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert('RGB'))

def image_to_tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)


class Occluder:
    def __init__(self,occluder_model_path) -> None:
        self.occluder_model_path = occluder_model_path
        self.face_occluder = self.get_face_occluder()


    def apply_execution_provider_options(self, execution_device_id: str, execution_providers: List[str]) -> List[Any]:
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

    def get_face_occluder(self):
        return onnxruntime.InferenceSession(self.occluder_model_path, providers=self.apply_execution_provider_options('0', ["CUDAExecutionProvider"]))

    def create_occlusion_mask(self,crop_vision_frame ):        
        prepare_vision_frame = cv2.resize(crop_vision_frame, self.face_occluder.get_inputs()[0].shape[1:3][::-1])
        prepare_vision_frame = np.expand_dims(prepare_vision_frame, axis = 0).astype(np.float32) / 255
        prepare_vision_frame = prepare_vision_frame.transpose(0, 1, 2, 3)
        occlusion_mask  = self.face_occluder.run(None, {
        self.face_occluder.get_inputs()[0].name: prepare_vision_frame
        })[0][0]
        occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
        occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return occlusion_mask


NODE_NAME = 'CreateFaceMask'


class FaceOcclusionModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "occluder_model_path": ("STRING", {"default": r"D:\AI\facestudio\res\models\face_occluder.onnx"}),
            }
        }

    RETURN_TYPES = ("FaceOcclusion_MODEL", )
    FUNCTION = "load_model"
    CATEGORY = "FoxTools"

    def load_model(self, occluder_model_path: str):
        model = Occluder(occluder_model_path)
        return (model, )

class CreateFaceMask:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_occluder_model": ("FaceOcclusion_MODEL", ),
                "input_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Mask",)
    FUNCTION = "generate_mask"
    CATEGORY = "FoxTools"

    def generate_mask(self, face_occluder_model, input_image):
        input_image = tensor_to_image(input_image[0])
        image_np = np.array(input_image)
        occlusion_mask = face_occluder_model.create_occlusion_mask(image_np)
        image_to = torch.from_numpy(occlusion_mask).unsqueeze(0)
        return (image_to, )

NODE_CLASS_MAPPINGS = {
    "FoxTools: FaceOcclusionModelLoader": FaceOcclusionModelLoader,
    "FoxTools: CreateFaceMask": CreateFaceMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoxTools: FaceOcclusionModelLoader": "FoxTools: Load Face Occlusion Model",
    "FoxTools: CreateFaceMask": "FoxTools: Create Face Mask"
}