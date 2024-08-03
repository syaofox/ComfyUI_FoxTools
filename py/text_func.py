import re

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

NODE_CLASS_MAPPINGS = {
    "Foxtools: RegTextFind": RegTextFind
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Foxtools: RegTextFind": "Foxtools: RegTextFind"
}


