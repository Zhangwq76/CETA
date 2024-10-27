import argparse
import itertools
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

tbwriter = SummaryWriter(log_dir=os.path.join('log', 'loss_768_16token'))


def read_mask(image_id, part_id, label) -> np.ndarray:
    if label == "body piece":
        mask_path = "/data/zhangxujie/dataset/farfetch512/train_big/seg/id_" + image_id + "/id_" + image_id + "_" + part_id + "_body_piece" + ".bmp"
    elif label == "band":
        mask_path = "/data/zhangxujie/dataset/farfetch512/train_big/seg/id_" + image_id + "/id_" + image_id + "_" + part_id + "_waist_band" + ".bmp"
    else:
        mask_path = "/data/zhangxujie/dataset/farfetch512/train_big/seg/id_" + image_id + "/id_" + image_id + "_" + part_id + "_" + label + ".bmp"

    mask = cv2.imread(mask_path)
    if mask is not None:
        return mask[..., 0:1]
    else:
        return np.zeros((512, 512, 1)).astype(np.uint8)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05,
                 image_root_path=""):
        super().__init__()
        # 预处理
        self.tokenizer = tokenizer
        self.clip_image_processor = CLIPImageProcessor()

        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        # self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]
        file = open(json_file, 'r', encoding='utf-8')
        data_ = []
        for line in file.readlines():
            dic = json.loads(line)
            data_.append(dic)
        self.data = data_

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _process_part_caption_image(self, image_id, part_segments, max_part_num=4, max_text_input_ids_num=4,
                                    mask_ratio=0.3, flip_ratio=0.3, i_drop_rate=0.05, t_drop_rate=0.05,):
        # Merge Same Part
        part_dict = {}
        for part in part_segments:
            part_label = part["coco_label"]
            if part_label in part_dict:
                part_dict[part_label].append(part["id"])
            else:
                part_dict[part_label] = [part["id"]]

        # Randomly select max_part_num parts
        selected_parts = random.sample(part_dict.keys(), min(max_part_num, len(part_dict)))

        # Part Text
        part_text_list = ["" if random.random() < t_drop_rate else part for part in selected_parts]
        part_text_list = part_text_list + [""] * (max_part_num - len(part_text_list))
        part_input_ids_list = self.tokenizer(
            part_text_list,
            max_length=max_text_input_ids_num,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).input_ids

        # Part Image
        image_path = self.image_root_path + "/img/id_" + image_id + ".jpg"
        raw_image = cv2.imread(image_path)
        part_image_list = []
        for part_label in selected_parts:
            if random.random() > i_drop_rate:
                # Get Mask
                part_ids = part_dict[part_label]
                masks = [read_mask(image_id, part_id, part_label) for part_id in part_ids]
                part_mask = np.sum(masks, axis=0)
                part_mask = (part_mask >= 255).astype(np.uint8)
                # Image Degrade
                degrade_image = raw_image.copy()
                #   1) Contrast and Lightness Control
                alpha = random.uniform(0, 5)  # Contrast Control
                beta = random.uniform(0, 50)  # Lightness Control
                degrade_image = cv2.convertScaleAbs(degrade_image, alpha=alpha, beta=beta)
                #   2) random set pixel to 0
                random_mask = np.random.rand(*part_mask.shape) < mask_ratio
                # repeat along last axis 3
                random_mask = np.repeat(random_mask, 3, axis=2)
                # print(random_mask.shape)

                degrade_image[random_mask] = 0
                #   3) Horizontal Flip
                if random.random() < flip_ratio:
                    degrade_image = cv2.flip(degrade_image, 1)
                    part_mask = cv2.flip(part_mask, 1)
                # Merge
                degrade_image[part_mask] = raw_image[part_mask]
            else:
                degrade_image = np.zeros((self.size, self.size, 3)).astype(np.uint8)
            # Transform
            degrade_image = Image.fromarray(cv2.cvtColor(degrade_image, cv2.COLOR_BGR2RGB))
            degrade_image = self.clip_image_processor(images=degrade_image, return_tensors="pt").pixel_values
            part_image_list.append(degrade_image)
        if len(part_image_list) < max_part_num:
            part_image_list = part_image_list + [torch.zeros(1, 3, 224, 224)] * (max_part_num - len(part_image_list))
        return part_input_ids_list, part_image_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # TODO: 每个衣服，最多 4或8 个 Part，可以先用 4 个试试，多的截断，少的补空✔
        #  每个部件的 text_token 长度限制在 4 个， 然后把他们拼成 [num_objects, 4] 的 tensor✔
        #  每个部件的 image 不要只给该部件的，可以随机多选1～n 个部件放进来做增强训练//咋放

        max_part_num, max_text_input_ids_num = 4, 4  # 每个部件的 text_token 长度限制在 4 个, 采用固定长度的 text_token
        item = self.data[idx]

        # Origin Image
        image_id = item['image_id']
        image_path = self.image_root_path + "/img/id_" + image_id + ".jpg"
        raw_image = Image.open(image_path).convert("RGB")
        image = self.transform(raw_image)
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        # Part Text & Image
        part_input_ids_list, part_clip_image_list = self._process_part_caption_image(image_id, item['segments'],
                                                                                     max_part_num,
                                                                                     max_text_input_ids_num)
        object_pixel_values = torch.cat([img for img in part_clip_image_list], dim=0)  # [4,3,224,224]

        return {
            "image": image,
            "clip_image": clip_image,
            "text_input_ids": part_input_ids_list,
            "txt_input_ids": torch.zeros(1, 77),
            # "drop_image_embed": drop_image_embed,
            "object_pixel_values": object_pixel_values
        }

        # segments = item['segments']
        #
        # max_num_objects = 4
        #
        # if max_num_objects > 2:
        #     # encoder_hidden_states = self.text_encoder(text_input_ids)
        #     text_path = self.image_root_path + "/txt/id_" + image_id + ".txt"
        #     # read image
        #     # raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        #     # image = self.transform(raw_image.convert("RGB"))
        #     # clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        #
        #     txt = ""
        #     # with open(text_path, "r") as f:
        #     #     txt = f.read()
        #     txt_input_ids = self.tokenizer(
        #         txt,
        #         max_length=77,
        #         padding='max_length',
        #         truncation=True,
        #         return_tensors="pt"
        #     ).input_ids  # [1,77]
        #
        #     # mask=[read_mask(seg["id"],seg["coco_label"]) for seg in segments]
        #     # part_clip_image=[get_part_clip_image(raw_image,item) for item in mask]
        #
        #     part_img = []
        #     image_ = cv2.imread(image_path)
        #     cloth_part_num = len(segments)
        #     for i in range(max_part_num):
        #         # print(i,image_id,len(segments))
        #         # preserve_part=np.zeros((512,512,3)).astype(np.uint8)
        #         seg = segments[i]
        #         name = 'image/' + image_id + '_' + seg["id"] + '_' + seg["coco_label"] + '.jpg'
        #         main_part_mask = read_mask(seg["id"], seg["coco_label"])
        #         main_part_mask = (main_part_mask == 255).astype(np.uint8)
        #         main_part = image_ * main_part_mask
        #         garment = main_part
        #         list_part = segments.copy()
        #         list_part.pop(i)
        #         for part in list_part:
        #             part_mask = read_mask(part["id"], part["coco_label"])
        #             part_mask = (part_mask == 255).astype(np.uint8)
        #             part_preserve = image_ * part_mask
        #             image_convert = image_.copy()
        #             if random.random() < 0.8:
        #                 alpha = random.uniform(1, 5)  # 对比度控制
        #                 beta = random.uniform(1, 50)  # 亮度控制
        #                 image_convert = cv2.convertScaleAbs(image_convert, alpha=alpha, beta=beta)
        #                 part_preserve = image_convert * part_mask
        #             # if random.random()< 0.3 and part["coco_label"]!=seg["coco_label"]:
        #             #     part_preserve = cv2.flip(part_preserve, 1)
        #             if random.random() < 0.1 or part["coco_label"] == "dress" or part["coco_label"] == "pants":  # drop
        #                 part_preserve = np.zeros((512, 512, 3)).astype(np.uint8)
        #             garment = garment + part_preserve
        #
        #         # extra_num = random.randint(0, cloth_part_num - part_num)
        #         # if extra_num>0:
        #         #     extra_mask_seg = random.sample(segments[part_num:cloth_part_num], extra_num)
        #         #     for seg in extra_mask_seg:
        #         #         mask = read_mask(seg["id"],seg["coco_label"])
        #         #         if random.random()< 0.1:
        #         #             mask = cv2.flip(mask, mode=1)
        #         #         main_part_mask = main_part_mask + mask
        #         # main_mask = (main_part_mask==255).astype(np.uint8)
        #         # main_part_mask = main_mask
        #         # garment=image_*main_part_mask
        #
        #         garment = Image.fromarray(cv2.cvtColor(garment, cv2.COLOR_BGR2RGB))
        #         # cv2.imwrite(name, garment)
        #         # garment.save(name)
        #         part_img.append(garment)
        #
        #     part_clip_image = [get_part_clip_image(part) for part in part_img]
        #
        #     # drop
        #     drop_image_embed = 0
        #     rand_num = random.random()
        #     if rand_num < self.i_drop_rate:
        #         drop_image_embed = 1
        #     elif rand_num < (self.i_drop_rate + self.t_drop_rate):
        #         text = ""
        #     elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
        #         text = ""
        #         drop_image_embed = 1
        #
        #     # id_list=[0]*max_num_objects
        #     # # print(max_num_objects)
        #     # for seg in segments:
        #     #     end=seg["end"]
        #     #     list_short=text_origin[0:end+1]
        #     #     n=2*list_short.count(",")-1
        #     #     # print(n)
        #     #     id_list[n]=1
        #
        #     object_pixel_values = torch.cat([img for img in part_clip_image], dim=0)  # [4,3,224,224]
        #     # print(object_pixel_values.shape)
        #     return {
        #         "image": image,
        #         "clip_image": clip_image,
        #         "text_input_ids": text_input_ids,
        #         "txt_input_ids": txt_input_ids,
        #         "drop_image_embed": drop_image_embed,
        #         "object_pixel_values": object_pixel_values
        #     }



def collate_fn(data):
    images = torch.stack([example["image"] for example in data])  # [8,4]
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)  # [8,3,224,224]
    text_input_ids = torch.stack([example["text_input_ids"] for example in data])  # [8,4,4]
    txt_input_ids = torch.stack([example["txt_input_ids"] for example in data])  # [8,1,77]
    # drop_image_embeds = [example["drop_image_embed"] for example in data]
    # object_pixel_values = [example["object_pixel_values"] for example in data] 
    object_pixel_values = torch.stack([example["object_pixel_values"] for example in data])  # [8,4,3,224,224]

    return {
        "images": images,
        "clip_images": clip_images,
        "text_input_ids": text_input_ids,
        "txt_input_ids": txt_input_ids,
        # "drop_image_embeds": drop_image_embeds,
        "object_pixel_values": object_pixel_values
    }


# FIXME: 下面的 FeatureBlender 就是原来的 CrossAttention, 增加了tensor 并行 和 MLP
#  把这个模块先看懂，然后改 main 函数 的代码，还有数据集也需要修改
from einops import rearrange


class FeatureBlender(nn.Module):
    def __init__(self,
                 num_output_tokens=4,
                 num_text_tokens=4,
                 num_objects=8,
                 text_token_dim=1024,
                 image_token_dim=1024,
                 cross_attention_dim=1024,
                 num_heads=8,
                 dropout=0.1,
                 ):
        super().__init__()
        self.heads = num_heads
        self.num_output_tokens = num_output_tokens
        self.num_objects = num_objects
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim

        self.to_query = nn.Linear(text_token_dim, cross_attention_dim)
        self.to_key = nn.Linear(image_token_dim, cross_attention_dim)
        self.to_value = nn.Linear(image_token_dim, cross_attention_dim)

        assert (num_objects * num_text_tokens * cross_attention_dim) % num_output_tokens == 0, \
            "num_output_tokens must be a divisor of num_objects * num_text_tokens * cross_attention_dim"
        self.to_out = nn.Sequential(
            # nn.Linear((num_objects * num_text_tokens * cross_attention_dim) // num_output_tokens, text_token_dim),
            nn.Linear((num_objects * num_text_tokens * cross_attention_dim) // num_output_tokens, text_token_dim),
            nn.GELU(),
            nn.Linear(text_token_dim, text_token_dim),
            nn.LayerNorm(text_token_dim),
            nn.Dropout(dropout)
        )

    def forward(self, image_features, text_embeds):
        """
        image_features: [batch_size, num_objects, num_image_tokens, image_token_dim] [8,4,4,1024]
        text_embeds: [batch_size, num_objects, num_text_tokens, text_token_dim] [8,4,4,768]
        return: [batch_size, num_output_tokens, text_token_dim]
        """
        b, _, _, _ = image_features.shape
        queries = self.to_query(
            text_embeds)  # [batch_size, num_objects, num_text_tokens, cross_attention_dim][40,4,4,1024]
        keys = self.to_key(
            image_features)  # [batch_size, num_objects, num_image_tokens, cross_attention_dim][40,4,64,1024]
        values = self.to_value(
            image_features)  # [batch_size, num_objects, num_image_tokens, cross_attention_dim][40,4,64,1024]

        # FIXME: 并行处理的原理是先把 部件维度 和 batch 维度合并，然后再拆分
        # To [batch_size * num_objects, num_tokens, num_heads, cross_attention_dim // num_heads]
        queries, keys, values = map(lambda x: rearrange(x, 'b n t (h d) -> (b n) t h d', h=self.heads).transpose(1, 2),
                                    (queries, keys, values))
        # print(queries.shape,keys.shape,values.shape)

        # Need PyTorch 2.0+
        assert is_torch2_available(), "This module requires PyTorch 2.0+"
        hidden_states = F.scaled_dot_product_attention(
            queries, keys, values, dropout_p=self.dropout, is_causal=False
        )  # [160,8,4,128]
        hidden_states = rearrange(hidden_states, '(b n) t h d -> b n t (h d)', b=b)
        # To [batch_size, num_output_tokens, cross_attention_dim]
        hidden_states = hidden_states.view(b, self.num_output_tokens,
                                           -1)  # [batch_size, num_output_tokens, num_object*num_text_tokens*1024]
        hidden_states = hidden_states.to(queries.dtype)

        # To out MLP
        out = self.to_out(hidden_states)  # [batch_size, num_output_tokens, text_token_dim][bsz,tokens,1024]
        # print("out:",out.shape)
        return out


# class CrossAttention(nn.Module):
#     def __init__(self, dim, heads = 8):
#         super().__init__()
#         self.dim = dim
#         self.heads = heads
#         self.scale = dim ** -0.5
#         self.query = nn.Linear(768, dim)
#         self.key = nn.Linear(dim, dim)
#         self.value = nn.Linear(dim, dim)
#
#     def forward(self, queries, keys, values, mask = None):
#         # print("queries:",queries.shape)
#         b, n, _, h = *queries.shape, self.heads
#         b_k, n_k, _, h = *keys.shape, self.heads
#         queries = self.query(queries)
#         keys = self.key(keys)
#         values = self.value(values)
#         queries = queries.view(b, n, h, -1).transpose(1, 2)
#         keys = keys.view(b_k, n_k, h, -1).transpose(1, 2)
#         values = values.view(b, n, h, -1).transpose(1, 2)
#         # print("queries:",queries.shape,"keys:",keys.shape)
#
#         dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale
#
#         if mask is not None:
#             mask = F.pad(mask.flatten(1), (1, 0), value = True)
#             assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
#             mask = mask[:, None, :].expand(-1, n, -1)
#             dots.masked_fill_(~mask, float('-inf'))
#
#         attn = dots.softmax(dim=-1)
#         out = torch.einsum('bhij,bhjd->bhid', attn, values)
#         out = out.transpose(1, 2).contiguous().view(b_k, n_k, -1)
#         return out

# FIXME: 下面的 GarmentComposer 是根据原来的 ip_adapter 的代码改的

class GarmentComposer(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, image_proj_model, adapter_modules, feature_blender, ckpt_path=None):
        super().__init__()
        self.image_proj_model = image_proj_model
        self.feature_blender = feature_blender
        self.adapter_modules = adapter_modules

        # if ckpt_path is not None:
        #     self.load_from_checkpoint(ckpt_path)

    def forward(self, image_embeds, text_embeds):
        """
        noisy_latents: [batch_size, 4, width, height]
        timesteps: int
        encoder_hidden_states: [batch_size, 77, 768]
        image_embeds: [batch_size, num_objects, num_image_tokens, image_token_dim] [8,4,1024]
        text_embeds: [batch_size, num_objects, num_text_tokens, text_token_dim] [8,4,4,768]
        """
        # image_embeds = rearrange(image_embeds, 'b n t d -> (b n) t d')
        image_tokens = self.image_proj_model(image_embeds)  # [batch_size * num_objects, 4, 768][8,4,4,768]
        # image_tokens = rearrange(image_tokens, '(b n) t d -> b n t d', b=image_embeds.shape[0])  # [batch_size, num_objects, 4, 768]
        blended_features = self.feature_blender(image_tokens, text_embeds)  # [9,1,1024]
        # print(blended_features.shape)
        # encoder_hidden_states = torch.cat([encoder_hidden_states, blended_features], dim=1) #[9,81,1024]
        # encoder_hidden_states = torch.cat([torch.zeros(2,77,768).to(blended_features.device), blended_features], dim=1)
        # Predict the noise residual
        # noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample #[9,4,1024]
        noise_pred = blended_features
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        orig_FeatureBlender_sum = torch.sum(torch.stack([torch.sum(p) for p in self.feature_blender.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)
        self.feature_blender.load_state_dict(state_dict["feature_blender"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        new_FeatureBlender_sum = torch.sum(torch.stack([torch.sum(p) for p in self.feature_blender.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_FeatureBlender_sum != new_FeatureBlender_sum, "Weights of feature_blender did not change!"
        # assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        # default=None,
        default="/data/zhangxujie/zwq/models/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default="/data/zhangxujie/zwq/models/ip-adapter_sd15.bin",
        # default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default="/data/zhangxujie/zwq/IP-Adapter-main/data_new.json",
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="/data/zhangxujie/dataset/farfetch512/train_big",
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="/data/zhangxujie/zwq/models/CLIP-ViT-H-14",
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter_fb_strength_16tokens_768",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="log/checkpoints",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,  # 1e-4
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=50, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,  # 8
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    state_dict = torch.load(args.pretrained_ip_adapter_path, map_location="cpu")
    ip_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    ip_proj_model.load_state_dict(state_dict["image_proj"], strict=True)


    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    # vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    ip_proj_model.requires_grad_(False)



    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    # feature_blender
    feature_blender = FeatureBlender(
        num_output_tokens=4,
        num_text_tokens=4,
        num_objects=4,
        text_token_dim=text_encoder.config.hidden_size,  # text_encoder.config.hidden_size
        image_token_dim=unet.config.cross_attention_dim,
        cross_attention_dim=unet.config.cross_attention_dim,
        num_heads=8,
        dropout=0.1,
    )
    # image feature projection model
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=16,
    )
    garment_composer = GarmentComposer(image_proj_model, adapter_modules, feature_blender,
                                       args.pretrained_ip_adapter_path)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    ip_proj_model.to(accelerator.device, dtype=weight_dtype)

    # optimizer
    # print(garment_composer)
    params_to_opt = itertools.chain(
        garment_composer.image_proj_model.parameters(),
        # garment_composer.adapter_modules.parameters(),
        garment_composer.feature_blender.parameters(),
    )
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution,
                              image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    garment_composer, optimizer, train_dataloader = accelerator.prepare(garment_composer, optimizer, train_dataloader)

    global_step = 0
    sum_avg_loss = 0
    total_steps = len(train_dataloader) * args.num_train_epochs
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_main_process)
    bsz = args.train_batch_size
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        epoch_loss = 0
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(garment_composer):
                # Convert images to latent space
                # with torch.no_grad():
                # latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                # latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                # noise = torch.randn_like(latents)
                # bsz = latents.shape[0]
                # Sample a random timestep for each image
                # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=accelerator.device)
                # timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    # encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0] 
                    # TODO: 按照 FeatureBlender 的并行方式做文本 encode并行加速✔
                    bsz, _, _ = batch["text_input_ids"].shape  # [8,4,4]
                    input_ids = rearrange(batch["text_input_ids"], 'b n t -> (b n) t')  # [32,4]
                    text_embeds = text_encoder(input_ids.to(accelerator.device))[0]  # [32,4,768]
                    text_embeds = rearrange(text_embeds, '(b n) t d -> b n t d', b=bsz)  # [8,4,4,768]
                    # encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0] 

                    # txt_ids = rearrange(batch["txt_input_ids"], 'b n t -> (b n) t')  # [8,1,77]->[8,77]
                    # encoder_hidden_states = text_encoder(txt_ids.to(accelerator.device))[0]  # [8,77,768]
                    # encoder_hidden_states = rearrange(encoder_hidden_states, '(b n) t d -> b n t d', b=bsz) #[8,4,77,768]

                with torch.no_grad():
                    # TODO: 按照 FeatureBlender 的并行方式做并行加速✔
                    bsz, _, _, _, _ = batch["object_pixel_values"].shape  # [8,4,3,224,224]
                    object_pixel_values = rearrange(batch["object_pixel_values"],
                                                    'b n c i j -> (b n) c i j')  # [32,3,224,224]
                    part_embeds = image_encoder(
                        object_pixel_values.to(accelerator.device, dtype=weight_dtype)).image_embeds  # [32,1024]
                    part_embeds = rearrange(part_embeds, '(b n) d -> b n d', b=bsz)  # [8,4,1024]

                    image_embeds = part_embeds
                    # print(image_embeds.shape)
                    # print("image_embeds",image_embeds.shape)
                ##drop
                # image_embeds_ = []
                # for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                #     if drop_image_embed == 1:
                #         image_embeds_.append(torch.zeros_like(image_embed))
                #     else:
                #         image_embeds_.append(image_embed)
                # image_embeds = torch.stack(image_embeds_)


                with torch.no_grad():
                    noise = image_encoder(
                        batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds  # [8,1024]
                    noise = rearrange(noise, 'b d -> b 1 d')  # [8,1,1024]
                    noise = ip_proj_model(noise) # [8,4,1024]
                    noise = noise.squeeze(1)  # [8,4,1024]
                    # print(noise_pred.shape, noise.shape)

                noise_pred = garment_composer(image_embeds, text_embeds)  # [8,1,1024]
                # print(noise_pred.shape, noise.shape)
                # noise = noise.unsqueeze(1)  # [8,1,1024]
                # noise = noise.repeat_interleave(4, dim=1)#[8,4,1024]

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")  # denoise_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                progress_bar.set_postfix({"loss": float(avg_loss)})
                sum_avg_loss = sum_avg_loss + avg_loss
                epoch_loss = epoch_loss + avg_loss
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                progress_bar.update(1)

                if accelerator.is_main_process:
                    # print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                    #     epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
                    if global_step % 100 == 0:
                        avg = sum_avg_loss / 100
                        tbwriter.add_scalar('loss', avg, global_step)
                        sum_avg_loss = 0

            if accelerator.is_main_process:
                avg_loss = epoch_loss / len(train_dataloader)
                tbwriter.add_scalar('epoch_loss', avg_loss, epoch)


            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, safe_serialization=False)

            begin = time.perf_counter()


if __name__ == "__main__":
    main()
