# Read_weights
Some tricks and basics of weights in AI models

## 2 ways of print the weights
1. directly key-weight pair
  - code: check_weights.py
    ```
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
    ```
  - results: anytext_key_weight_pair.json
    ```
    ...
    {
        "posterior_mean_coef2": [
            1000
        ]
    },
    {
        "logvar": [
            1000
        ]
    },
    {
        "model.diffusion_model.time_embed.0.weight": [
            1280,
            320
        ]
    },
    {
        "model.diffusion_model.time_embed.0.bias": [
            1280
        ]
    },
    {
        "model.diffusion_model.time_embed.2.weight": [
            1280,
            1280
        ]
    },...
    ```
2. nested_dict
   - code: check_weights_pair.py
    ```
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
    ```
  - results: anytext_v1.1.json
    ```
    ...
      "posterior_mean_coef1": [
        1000
    ],
    "posterior_mean_coef2": [
        1000
    ],
    "logvar": [
        1000
    ],
    "model": {
        "diffusion_model": {
            "time_embed": {
                "0": {
                    "weight": [
                        1280,
                        320
                    ],
                    "bias": [
                        1280
                    ]
                },
                "2": {
                    "weight": [
                        1280,
                        1280
                    ],
                    "bias": [
                        1280
                    ]
                }
            },
            "input_blocks": {
                "0": {
                    "0": {
                        "weight": [
                            320,
                            4,
                            3,
                            3
                        ],
                        "bias": [
                            320
                        ]
                    }
                },
    ...
    ```
