import re
from pathlib import Path

class RegTextFind:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":  ("STRING", {"default": '', "multiline": False}),
                "pattern": ("STRING", {"default": '', "multiline": False}),
                "group": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"

    CATEGORY = "FoxTools/Text"

    def execute(self, text, pattern, group):

        result = ''
        match = re.search(pattern, text)

        if match:
            result = match.group(group)
            print(result)  
        else:
            print("No match found")

        return (result,)
    

class GenSwapPathText:
    dir_dict = {}

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        target_dir =r"D:\adult\__face_pices\group"
        for d in  Path(target_dir).iterdir():
            if d.is_dir():
                cls.dir_dict[d.name] = d
        


        return {
            "required": {
                "face_name": (list(cls.dir_dict.keys()),),    
                "pices_path":  ("STRING", {"default": '', "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("face_path", "pices_path", "saves_path",)
    FUNCTION = "execute"

    CATEGORY = "FoxTools/Text"

    def execute(self, face_name, pices_path):

 
        pices_driname = Path(pices_path).name
        saves_path = rf'{face_name}\{pices_driname}\swap_'

        return (str(self.dir_dict[face_name]), pices_path, saves_path,)
    


NODE_CLASS_MAPPINGS = {
    "Foxtools: RegTextFind": RegTextFind,
    "Foxtools: GenSwapPathText": GenSwapPathText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Foxtools: RegTextFind": "Foxtools: RegTextFind",
    "Foxtools: GenSwapPathText": "Foxtools: GenSwapPathText"
}


