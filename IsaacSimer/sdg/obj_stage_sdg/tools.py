import json
import math


def load_config(jsonPath):
    with open(jsonPath, "r", encoding="utf-8") as file:
        data = json.load(file)

    return convert_string_bools(data)


def convert_string_bools(obj):
    """
    递归遍历数据结构，将字符串 "False" 和 "True" 转换为 Python 的布尔值。
    同样也可以处理 "None" 或其他需要转换的字符串。
    """
    if isinstance(obj, dict):
        # 如果是字典，遍历每个键值对
        for key, value in obj.items():
            obj[key] = convert_string_bools(value)
        return obj
    elif isinstance(obj, list):
        # 如果是列表，遍历每个元素
        return [convert_string_bools(item) for item in obj]
    elif isinstance(obj, str):
        # 如果是字符串，检查是否是特定的需要转换的值
        if obj.lower() == "false":  # 处理大小写可能不一致的情况
            return False
        elif obj.lower() == "true":
            return True
        elif obj.lower() == "none" or obj.lower() == "null":
            return None
        # 添加其他你需要的字符串转换规则...
        else:
            return obj
    else:
        # 如果不是字典、列表或字符串，直接返回（如数字、真正的布尔值等）
        return obj
