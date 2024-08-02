IS_DLIB_INSTALLED = False
try:
    import dlib
    IS_DLIB_INSTALLED = True
except ImportError:
    pass

IS_INSIGHTFACE_INSTALLED = False
try:
    from insightface.app import FaceAnalysis
    IS_INSIGHTFACE_INSTALLED = True
except ImportError:
    pass

if not IS_DLIB_INSTALLED and not IS_INSIGHTFACE_INSTALLED:
    raise Exception("Please install either dlib or insightface to use this node.")

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from comfy.utils import ProgressBar
import os
import folder_paths
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageOps, ImageFilter

# DLIB_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dlib")
DLIB_DIR = os.path.join(folder_paths.models_dir, "dlib")
INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

THRESHOLDS = { # from DeepFace
        "VGG-Face": {"cosine": 0.68, "euclidean": 1.17, "L2_norm": 1.17},
        "Facenet": {"cosine": 0.40, "euclidean": 10, "L2_norm": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "L2_norm": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "L2_norm": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "L2_norm": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "L2_norm": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "L2_norm": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "L2_norm": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "L2_norm": 0.17},
        "GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "L2_norm": 1.10},
    }

def tensor_to_image(image):
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert('RGB'))

def image_to_tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)
    #return T.ToTensor()(Image.fromarray(image)).permute(1, 2, 0)

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

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R, 
                                 c2.T - (s2 / s1) * R * c1.T)),
                                 np.matrix([0., 0., 1.])])

def mask_from_landmarks(image, landmarks):
    import cv2

    mask = np.zeros(image.shape[:2], dtype=np.float64)
    points = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, points, color=1)

    return mask

class InsightFace:
    def __init__(self, provider="CPU", name="buffalo_l"):
        self.face_analysis = FaceAnalysis(name=name, root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        self.thresholds = THRESHOLDS["ArcFace"]

    def get_face(self, image):
        for size in [(size, size) for size in range(640, 256, -64)]:
            self.face_analysis.det_model.input_size = size
            faces = self.face_analysis.get(image)
            if len(faces) > 0:
                return sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True)
        return None

    def get_embeds(self, image):
        face = self.get_face(image)
        if face is not None:
            face = face[0].normed_embedding
        return face
    
    def get_bbox(self, image, padding=0, padding_percent=0):
        faces = self.get_face(np.array(image))
        img = []
        x = []
        y = []
        w = []
        h = []
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            width = x2 - x1
            height = y2 - y1
            x1 = int(max(0, x1 - int(width * padding_percent) - padding))
            y1 = int(max(0, y1 - int(height * padding_percent) - padding))
            x2 = int(min(image.width, x2 + int(width * padding_percent) + padding))
            y2 = int(min(image.height, y2 + int(height * padding_percent) + padding))
            crop = image.crop((x1, y1, x2, y2))
            img.append(T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0))
            x.append(x1)
            y.append(y1)
            w.append(x2 - x1)
            h.append(y2 - y1)
        return (img, x, y, w, h)
    
    def get_keypoints(self, image):
        face = self.get_face(image)
        if face is not None:
            shape = face[0]['kps']
            right_eye = shape[0]
            left_eye = shape[1]
            nose = shape[2]
            left_mouth = shape[3]
            right_mouth = shape[4]
            
            return [left_eye, right_eye, nose, left_mouth, right_mouth]
        return None

    def get_landmarks(self, image, extended_landmarks=False):
        face = self.get_face(image)
        if face is not None:
            shape = face[0]['landmark_2d_106']
            landmarks = np.round(shape).astype(np.int64)

            main_features = landmarks[33:]
            left_eye = landmarks[87:97]
            right_eye = landmarks[33:43]
            eyes = landmarks[[*range(33,43), *range(87,97)]]
            nose = landmarks[72:87]
            mouth = landmarks[52:72]
            left_brow = landmarks[97:106]
            right_brow = landmarks[43:52]
            outline = landmarks[[*range(33), *range(48,51), *range(102, 105)]]
            outline_forehead = outline

            return [landmarks, main_features, eyes, left_eye, right_eye, nose, mouth, left_brow, right_brow, outline, outline_forehead]
        return None

class DLib:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        # check if the models are available
        if not os.path.exists(os.path.join(DLIB_DIR, "shape_predictor_5_face_landmarks.dat")):
            raise Exception("The 5 point landmark model is not available. Please download it from https://huggingface.co/matt3ounstable/dlib_predictor_recognition/blob/main/shape_predictor_5_face_landmarks.dat")
        if not os.path.exists(os.path.join(DLIB_DIR, "dlib_face_recognition_resnet_model_v1.dat")):
            raise Exception("The face recognition model is not available. Please download it from https://huggingface.co/matt3ounstable/dlib_predictor_recognition/blob/main/dlib_face_recognition_resnet_model_v1.dat")

        self.shape_predictor = dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_5_face_landmarks.dat"))
        self.face_recognition = dlib.face_recognition_model_v1(os.path.join(DLIB_DIR, "dlib_face_recognition_resnet_model_v1.dat"))
        self.thresholds = THRESHOLDS["Dlib"]

    def get_face(self, image):
        faces = self.face_detector(np.array(image), 1)
        #faces, scores, _ = self.face_detector.run(np.array(image), 1, -1)
        
        if len(faces) > 0:
            return sorted(faces, key=lambda x: x.area(), reverse=True)
            #return [face for _, face in sorted(zip(scores, faces), key=lambda x: x[0], reverse=True)] # sort by score
        return None

    def get_embeds(self, image):
        faces = self.get_face(image)
        if faces is not None:
            shape = self.shape_predictor(image, faces[0])
            faces = np.array(self.face_recognition.compute_face_descriptor(image, shape))
        return faces
    
    def get_bbox(self, image, padding=0, padding_percent=0):
        faces = self.get_face(image)
        img = []
        x = []
        y = []
        w = []
        h = []
        for face in faces:
            x1 = max(0, face.left() - int(face.width() * padding_percent) - padding)
            y1 = max(0, face.top() - int(face.height() * padding_percent) - padding)
            x2 = min(image.width, face.right() + int(face.width() * padding_percent) + padding)
            y2 = min(image.height, face.bottom() + int(face.height() * padding_percent) + padding)
            crop = image.crop((x1, y1, x2, y2))
            img.append(T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0))
            x.append(x1)
            y.append(y1)
            w.append(x2 - x1)
            h.append(y2 - y1)
        return (img, x, y, w, h)
    
    def get_keypoints(self, image):
        faces = self.get_face(image)
        if faces is not None:
            shape = self.shape_predictor(image, faces[0])
          
            left_eye = [(shape.part(0).x + shape.part(1).x // 2), (shape.part(0).y + shape.part(1).y) // 2]
            right_eye = [(shape.part(2).x + shape.part(3).x // 2), (shape.part(2).y + shape.part(3).y) // 2]
            nose = [shape.part(4).x, shape.part(4).y]

            return [left_eye, right_eye, nose]
        return None
    
    def get_landmarks(self, image, extended_landmarks=False):
        if extended_landmarks:
            if not os.path.exists(os.path.join(DLIB_DIR, "shape_predictor_81_face_landmarks.dat")):
                raise Exception("The 68 point landmark model is not available. Please download it from https://huggingface.co/matt3ounstable/dlib_predictor_recognition/blob/main/shape_predictor_81_face_landmarks.dat")
            predictor = dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_81_face_landmarks.dat"))
        else:
            if not os.path.exists(os.path.join(DLIB_DIR, "shape_predictor_68_face_landmarks.dat")):
                raise Exception("The 68 point landmark model is not available. Please download it from https://huggingface.co/matt3ounstable/dlib_predictor_recognition/blob/main/shape_predictor_68_face_landmarks.dat")
            predictor = dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_68_face_landmarks.dat"))

        faces = self.get_face(image)
        if faces is not None:
            shape = predictor(image, faces[0])
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            main_features = landmarks[17:68]
            left_eye = landmarks[42:48]
            right_eye = landmarks[36:42]
            eyes = landmarks[36:48]
            nose = landmarks[27:36]
            mouth = landmarks[48:68]
            left_brow = landmarks[17:22]
            right_brow = landmarks[22:27]
            outline = landmarks[[*range(17), *range(26,16,-1)]]
            if extended_landmarks:
                outline_forehead = landmarks[[*range(17), *range(26,16,-1), *range(68, 81)]]
            else:
                outline_forehead = outline

            return [landmarks, main_features, eyes, left_eye, right_eye, nose, mouth, left_brow, right_brow, outline, outline_forehead]
        return None

    
class FaceAlignSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image_from": ("IMAGE", ),
            }, "optional": {
                "image_to": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", )
    FUNCTION = "align"
    CATEGORY = "FoxTools"

    def align(self, analysis_models, image_from, image_to=None):
        def closest_angle(angle):
        # List of target angles
            target_angles = [0, 90, 180, 270, 360]
            # Find the closest angle in the list
            closest = min(target_angles, key=lambda x: abs(x - angle))
            return closest

        image_from = tensor_to_image(image_from[0])
        shape = analysis_models.get_keypoints(image_from)
        
        l_eye_from = shape[0]
        r_eye_from = shape[1]
        angle = float(np.degrees(np.arctan2(l_eye_from[1] - r_eye_from[1], l_eye_from[0] - r_eye_from[0])))

        if image_to is not None:
            image_to = tensor_to_image(image_to[0])
            shape = analysis_models.get_keypoints(image_to)
            l_eye_to = shape[0]
            r_eye_to = shape[1]
            angle -= float(np.degrees(np.arctan2(l_eye_to[1] - r_eye_to[1], l_eye_to[0] - r_eye_to[0])))

        # Adjust angle to be the closest to 0, 90, 180, 270, 360
        angle = closest_angle(angle)

        # Rotate the image
        image_from = Image.fromarray(image_from).rotate(angle, expand=True)
        image_from = image_to_tensor(image_from).unsqueeze(0)

        return (image_from, 360-angle)
    

class FaceAlignCacul:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image_from": ("IMAGE", ),
            }, "optional": {
                "image_to": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("INT", "INT", )
    RETURN_NAMES = ("rotate","unrotate")
    FUNCTION = "align"
    CATEGORY = "FoxTools"

    def align(self, analysis_models, image_from, image_to=None):
        def closest_angle(angle):
            print(angle)
        # List of target angles
            if angle <0:
                angle = 360+angle

            target_angles = [0, 90, 180, 270, 360]
            # Find the closest angle in the list
            closest = min(target_angles, key=lambda x: abs(x - angle))
            return closest

        image_from = tensor_to_image(image_from[0])
        shape = analysis_models.get_keypoints(image_from)

        if shape is None:
            angle =90
            return (angle, 360-angle)
        
        l_eye_from = shape[0]
        r_eye_from = shape[1]
        angle = float(np.degrees(np.arctan2(l_eye_from[1] - r_eye_from[1], l_eye_from[0] - r_eye_from[0])))

        if image_to is not None:
            image_to = tensor_to_image(image_to[0])
            shape = analysis_models.get_keypoints(image_to)
            l_eye_to = shape[0]
            r_eye_to = shape[1]
            angle -= float(np.degrees(np.arctan2(l_eye_to[1] - r_eye_to[1], l_eye_to[0] - r_eye_to[0])))

        # Adjust angle to be the closest to 0, 90, 180, 270, 360
        angle = closest_angle(angle)

        # Rotate the image
        # image_from = Image.fromarray(image_from).rotate(angle, expand=True)
        # image_from = image_to_tensor(image_from).unsqueeze(0)

        return (angle, 360-angle)


class FaceBlurBord:
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

    RETURN_TYPES = ("IMAGE", )
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
    
       
        blurred_image = image_to_tensor(blurred_image).unsqueeze(0)


        return (blurred_image, )
    


NODE_CLASS_MAPPINGS = {  
    "SimpleFaceAlign": FaceAlignSimple,
    "CaculFaceAlign": FaceAlignCacul,
    "GenBlurBord": FaceBlurBord,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleFaceAlign": "Simple FaceAlign",
    "CaculFaceAlign": "Cacul FaceAlign",
    "GenBlurBord": "Gen Blurbord",
}
