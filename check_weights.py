import json
import torch
from safetensors.torch import load_file  # 导入 safetensors 库

def build_nested_dict(weights):
    """
    构建一个嵌套字典以反映权重的层级结构
    :param weights: 加载的权重
    :return: 嵌套字典
    """
    nested_dict = {}
    for key, value in weights.items():
        parts = key.split('.')
        d = nested_dict
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # 仅存储张量的形状
        d[parts[-1]] = list(value.size()) if isinstance(value, torch.Tensor) else value
    return nested_dict

# 读取权重
weights_path = '/home/stone/team/user/stone/models/custom-diff/1-26-2.8w/checkpoint-40000/pytorch_custom_diffusion_weights.safetensors'
# "/home/stone/.cache/modelscope/hub/damo/cv_anytext_text_generation_editing/anytext_v1.1.ckpt"
# weights = torch.load(weights_path, map_location='cpu')
weights = load_file(weights_path)  # 使用 safetensors 加载权重


# 构建层级结构的字典
nested_weights = build_nested_dict(weights)

# 将结果按原结构存为 json
output_json_path = "/home/stone/nas/AnyText/ckpt_format/pytorch_custom_diffusion_weights.json"
with open(output_json_path, 'w') as json_file:
    json.dump(nested_weights, json_file, indent=4)

# ... existing code ...