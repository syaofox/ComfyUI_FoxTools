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
    


class ShowText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"

    def notify(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}



NODE_CLASS_MAPPINGS = {
    "FoxRegTextFind": RegTextFind,
    "FoxGenSwapPathText": GenSwapPathText,
    "FoxShowText": ShowText,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoxRegTextFind": "Foxtools: RegTextFind",
    "FoxShowText": "Foxtools: ShowText",
}


