import cv2
import numpy as np
import onnxruntime
import torch
import torchvision.transforms.v2 as T
import torch.nn.functional as F
from typing import Any, List
from comfy.utils import ProgressBar


def tensor_to_image(image):
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert('RGB'))

def image_to_tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)



def expand_mask(mask, expand, tapered_corners):
    import scipy

    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    device = mask.device
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)

    return torch.stack(out, dim=0).to(device)



class Occluder:
    def __init__(self, occluder_model_path) -> None:
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

    def create_occlusion_mask(self, crop_vision_frame):
        prepare_vision_frame = cv2.resize(crop_vision_frame, self.face_occluder.get_inputs()[0].shape[1:3][::-1])
        prepare_vision_frame = np.expand_dims(prepare_vision_frame, axis=0).astype(np.float32) / 255
        prepare_vision_frame = prepare_vision_frame.transpose(0, 1, 2, 3)
        occlusion_mask = self.face_occluder.run(None, {
            self.face_occluder.get_inputs()[0].name: prepare_vision_frame
        })[0][0]
        occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
        occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return occlusion_mask

class FaceOcclusionModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "occluder_model_path": ("STRING", {"default": r"D:\AI\facestudio\res\models\face_occluder.onnx"}),
            }
        }

    RETURN_TYPES = ("FaceOcclusion_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "FoxTools"

    def load_model(self, occluder_model_path: str):
        model = Occluder(occluder_model_path)
        return (model,)


class CreateFaceMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_occluder_model": ("FaceOcclusion_MODEL",),
                "input_image": ("IMAGE",),
                "grow": ("INT", { "default": 0, "min": -4096, "max": 4096, "step": 1 }),
                "grow_tapered": ("BOOLEAN", { "default": False }),
                "blur": ("INT", { "default": 1, "min": 1, "max": 4096, "step": 2 }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", )
    RETURN_NAMES = ("mask", "image", )
    FUNCTION = "generate_mask"
    CATEGORY = "FoxTools"

    def generate_mask(self, face_occluder_model, input_image, grow, grow_tapered, blur):

        steps = input_image.shape[0]

        if steps > 1:
            pbar = ProgressBar(steps)

        out_mask = []
        out_image = []


        for img in input_image:
            face = tensor_to_image(img)

            if face is None:
                print(f"\033[96mNo face detected at frame {len(out_image)}\033[0m")
                img = torch.zeros_like(img)
                mask = img.clone()[:,:,:1]
                out_mask.append(mask)
                out_image.append(img)

                continue

            
            pil_image = tensor_to_image(img)
            cv2_image = np.array(pil_image)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)        
            occlusion_mask = face_occluder_model.create_occlusion_mask(cv2_image)

            if occlusion_mask is None:
                print(f"\033[96mNo landmarks detected at frame {len(out_image)}\033[0m")
                img = torch.zeros_like(img)
                mask = img.clone()[:,:,:1]
                out_mask.append(mask)
                out_image.append(img)

                continue


            mask = image_to_tensor(occlusion_mask).unsqueeze(0).squeeze(-1).clamp(0, 1).to(device=img.device)

            # _, y, x = torch.where(mask)
            # x1, x2 = x.min().item(), x.max().item()
            # y1, y2 = y.min().item(), y.max().item()
            # smooth = int(min(max((x2 - x1), (y2 - y1)) * 0.2, 99))

            # if smooth > 1:
            #     if smooth % 2 == 0:
            #         smooth+= 1
            #     mask = T.functional.gaussian_blur(mask.bool().unsqueeze(1), smooth).squeeze(1).float()
            
            if grow != 0:
                mask = expand_mask(mask, grow, grow_tapered)

            if blur > 1:
                if blur % 2 == 0:
                    blur+= 1
                mask = T.functional.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1).float()



            mask = mask.squeeze(0).unsqueeze(-1)


       
            img = img * mask.repeat(1, 1, 3)
            out_mask.append(mask)
            out_image.append(img)

            if steps > 1:
                pbar.update(1)       


        out_mask = torch.stack(out_mask).squeeze(-1)
        out_image = torch.stack(out_image)
        
        return (out_mask, out_image, )

    

NODE_CLASS_MAPPINGS = {
    "FoxTools: FaceOcclusionModelLoader": FaceOcclusionModelLoader,
    "FoxTools: CreateFaceMask": CreateFaceMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoxTools: FaceOcclusionModelLoader": "FoxTools: Load Face Occlusion Model",
    "FoxTools: CreateFaceMask": "FoxTools: Create Face Mask"
}
