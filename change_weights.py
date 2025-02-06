import torch
from safetensors.torch import load_file

# 定义对应关系
mapping = {
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight": "mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight": "mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_k_custom_diffusion.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_v_custom_diffusion.weight"
}

# 加载 anytext_v1.1.ckpt
original_weights = torch.load('/home/stone/.cache/modelscope/hub/damo/cv_anytext_text_generation_editing/anytext_v1.1.ckpt')

# 加载 pytorch_custom_diffusion_weights.safetensors
custom_diffusion_weights = load_file('/home/stone/team/user/stone/models/custom-diff/1-26-2.8w/checkpoint-40000/pytorch_custom_diffusion_weights.safetensors')

# 进行权重替换
for original_key, custom_key in mapping.items():
    if original_key in original_weights and custom_key in custom_diffusion_weights:
        original_weights[original_key] = custom_diffusion_weights[custom_key]
    else:
        print(f"Key {original_key} in original weights or {custom_key} in custom weights not found.")

# 保存新的权重文件
new_ckpt_path = '/home/stone/team/user/stone/models/custom-diff/anytext_cd/anytext_v1.1_cd.ckpt'
torch.save(original_weights, new_ckpt_path)
print(f"新的权重文件已保存到 {new_ckpt_path}")