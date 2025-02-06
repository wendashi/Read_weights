import torch
import json
from safetensors.torch import load_file  # 导入 safetensors 库


def get_weight_key_shape_pairs(data, prefix=""):
    pairs = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, torch.Tensor):
                shape = list(value.shape)
                pairs.append({new_prefix: shape})
            else:
                pairs.extend(get_weight_key_shape_pairs(value, new_prefix))
    elif isinstance(data, torch.Tensor):
        shape = list(data.shape)
        pairs.append({prefix: shape})
    return pairs


# 加载 .ckpt 文件
# checkpoint_path = "/home/stone/.cache/modelscope/hub/damo/cv_anytext_text_generation_editing/anytext_v1.1.ckpt"
# checkpoint = torch.load(checkpoint_path)

# 加载 .safetensors 文件
checkpoint_path = '/home/stone/team/user/stone/models/custom-diff/1-26-2.8w/checkpoint-40000/pytorch_custom_diffusion_weights.safetensors'
checkpoint = load_file(checkpoint_path)  

# 获取所有权重的键和形状对
key_shape_pairs = get_weight_key_shape_pairs(checkpoint)

# 将结果保存为 JSON 文件
output_json_path = "/home/stone/nas/AnyText/ckpt_format/pytorch_custom_diffusion_key_weight_pair.json"
with open(output_json_path, 'w') as f:
    json.dump(key_shape_pairs, f, indent=4)

print(f"结果已保存到 {output_json_path}")