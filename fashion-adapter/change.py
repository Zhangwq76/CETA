import torch
import safetensors
from safetensors import safe_open

# ckpt = "sd-ip_adapter/checkpoint-170000/model.safetensors"
# sd = safe_open(ckpt, framework="pt", device=0)
ckpt = "/data/zhangxujie/zwq/IP-Adapter-main/fb_strength_16token->1token_fashionsd_44w/checkpoint-27500/pytorch_model.bin"
sd = torch.load(ckpt, map_location="cpu")
image_proj_sd = {}
ip_sd = {}
feature_blender = {}

# with safe_open(ckpt, framework="pt", device=0) as f:
#     for k in f.keys():
#         if k.startswith("unet"):
#             pass
#         elif k.startswith("image_proj_model"):
#             image_proj_sd[k.replace("image_proj_model.", "")] = f.get_tensor(k)
#         elif k.startswith("adapter_modules"):
#             ip_sd[k.replace("adapter_modules.", "")] = f.get_tensor(k)
#         elif k.startswith("feature_blender"):
#             feature_blender[k.replace("feature_blender.", "")] = f.get_tensor(k)
            
for k in sd:
    if k.startswith("unet"):
        pass
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]
    elif k.startswith("feature_blender"):
        feature_blender[k.replace("feature_blender.", "")] = sd[k]

torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd, "feature_blender": feature_blender}, "fashion_fb_27500.bin")
# torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "ip_adapter_10000_FashionAdapter.bin")