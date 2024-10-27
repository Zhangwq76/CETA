# import spacy

# # Load English tokenizer, tagger, parser and NER
# nlp = spacy.load("en_core_web_sm")

# # Process whole documents
# text = ("Black/white logo-print long-sleeved hoodie from MONCLER GRENOBLE featuring logo print to the front, logo patch at the sleeve, drawstring hood, drop shoulder, long sleeves and straight hem.logo-print long-sleeved hoodie")
# doc = nlp(text)

# # Analyze syntax
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
# # print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# # # Find named entities, phrases and concepts
# # for entity in doc.ents:
# #     print(entity.text, entity.label_)

# import numpy as np
# import matplotlib.pyplot as plt
# depthmap = np.load('000000000.npy')    #使用numpy载入npy文件
# plt.imshow(depthmap)              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
# # plt.colorbar()                   #添加colorbar       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
# plt.savefig('depthmap.jpg')
# plt.show() 

import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import json
import random
# from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

# text_encoder = CLIPTextModel.from_pretrained("/data/zhangxujie/zwq/models/stable-diffusion-v1-5/text_encoder")
# tokenizer = CLIPTokenizer.from_pretrained("/data/zhangxujie/zwq/models/stable-diffusion-v1-5/tokenizer")


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


# mask111 = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/seg/id_000000001/id_000000001_2_sleeve.bmp")[...,0:1]
# cv2.imwrite('111.jpg', mask111)
# # ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# def mask_find_bboxs(mask):
#     retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
#     stats = stats[stats[:,4].argsort()]
#     return stats[:-1] 

# bboxs = mask_find_bboxs(mask)

# b=bboxs[-1]
# x0, y0 = b[0], b[1]
# x1 = b[0] + b[2]
# y1 = b[1] + b[3]
# # print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
# start_point, end_point = (x0, y0), (x1, y1)

# print(start_point,end_point)

# color = (0, 0, 255) # Red color in BGR；红色：rgb(255,0,0)
# thickness = 1 # Line thickness of 1 px 
# mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 转换为3通道图，使得color能够显示红色。
# mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)

# """
# # Displaying the image  
# cv2.imshow('show_image', mask_bboxs) 
# cv2.waitKey(0)
# """
# cv2.imwrite('./save.jpg', mask_bboxs)
# kind={}
savelist1=["piece.bmp"]
image_list_txt="/data/zhangxujie/dataset/farfetch512/train_big/train_big_new.txt" 

# f=open("train_data_big.json",'a')
n=0
dataroot="/data/zhangxujie/dataset/farfetch512/train_big/seg/"
txt_root="/data/zhangxujie/dataset/farfetch512/train_big/seg_txt/"
part_list=["body_part","collar","sleeve_type","pocket","waist","category"]

with open(image_list_txt, 'r') as f_image:
    lines = f_image.readlines()
for lin in lines: 
    dic={}
    caption_save=""
    mm=0
    dic["image_id"]=lin.split(" ")[0].split("_")[1].split(".")[0]

    for curDir, dirs, files in os.walk(dataroot+"id_"+dic["image_id"]):
        segments=[]
        # print(files)

        for file in files:
            if file.endswith(".bmp"):
                segment={}

                part=file.split('_')[-1]
                if part==savelist1[0]:   #这个部件在不在要留的表里
                    id=file.split('_')[2]
                    part_name='body'+' '+part.split('.')[0]
                # elif part=="pocket.bmp" and ("pocket" not in caption_save):
                #     caption_save = caption_save + "pocket"
                #     id=file.split('_')[2]
                #     part_name=part.split('.')[0]
                else:
                    id=file.split('_')[2]
                    part_name=part.split('.')[0]

                if part_name!="top":
                    start=len(caption_save)
                    caption_save=caption_save + part_name + ","
                    end=len(caption_save)-1

                    segment["id"]=id
                    segment["start"]=start
                    segment["end"]=end
                    # segment["bbox"]=bbox
                    segment["coco_label"]=part_name
                    # print(segment)
                    # print(file.split('_')[1],id,part_name,bbox)
                    segments.append(segment)
                    # if part_name in kind:
                    #     kind[part_name] = kind[part_name]+1
                    # else:
                    #     kind[part_name] = 1                    
    # n=n+1
    # print(n)
# print(kind)
    dic["caption"]=caption_save
    dic["segments"]=segments
    if dic["image_id"]=="000000084":
        print(dic["segments"])
        break
    # print(dic)
    # if len(dic["segments"])>=4:
    #     json_write = json.dumps(dic,cls=MyEncoder)
    #     f.write(json_write)
    #     f.write('\n')
    #     print(n)
    #     n=n+1
    #     if n==200000:
    #         break
# n=0
# kind = {}
# with open(image_list_txt, 'r') as f_image:
#     lines = f_image.readlines()
# for lin in lines: 
#     image_id=lin.split(" ")[0].split("_")[1].split(".")[0]
#     json_file = txt_root+"id_"+image_id+".json"
#     if os.path.exists(json_file):
#         with open(json_file,"r", encoding='utf-8'  ) as f:
#             line = f.readlines()[0]
#             js = json.loads(line)
#             if js is not None:
#                 if "category" in js:
#                     category=js["category"]
#                     if category in kind:
#                         kind[category] = kind[category]+1
#                     else:
#                         kind[category] = 1
#         # print(kind)
#     n=n+1
#     print(n)









# from transformers import CLIPImageProcessor
# from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
# from torchvision import transforms
# from PIL import Image
# from accelerate import Accelerator
# import torch.nn as nn
# import torch
# from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
# from transformers.models.clip.modeling_clip import (
#     CLIPTextTransformer,
#     CLIPPreTrainedModel,
#     CLIPModel,
# )
# from diffusers.models.attention_processor import (
#     Attention,
#     AttnProcessor,
#     AttnProcessor2_0,
# )
# layers=5
# UNET_LAYER_NAMES = [
#         "down_blocks.0",
#         "down_blocks.1",
#         "down_blocks.2",
#         "mid_block",
#         "up_blocks.1",
#         "up_blocks.2",
#         "up_blocks.3",
#     ]
# pretrained_model_name_or_path="/data1/chongzheng/Models/SD/stable-diffusion-v1-5"

# class MLP(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
#         super().__init__()
#         if use_residual:
#             assert in_dim == out_dim
#         self.layernorm = nn.LayerNorm(in_dim)
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, out_dim)
#         self.use_residual = use_residual
#         self.act_fn = nn.GELU()

#     def forward(self, x):
#         residual = x
#         x = self.layernorm(x)
#         x = self.fc1(x)
#         x = self.act_fn(x)
#         x = self.fc2(x)
#         if self.use_residual:
#             x = x + residual
#         return x

# class BalancedL1Loss(nn.Module):
#     def __init__(self, threshold=1.0, normalize=False):
#         super().__init__()
#         self.threshold = threshold
#         self.normalize = normalize

#     def forward(self, object_token_attn_prob, object_segmaps):
#         if self.normalize:
#             object_token_attn_prob = object_token_attn_prob / (
#                 object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5
#             )
#         background_segmaps = 1 - object_segmaps
#         background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
#         object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

#         background_loss = (object_token_attn_prob * background_segmaps).sum(
#             dim=2
#         ) / background_segmaps_sum

#         object_loss = (object_token_attn_prob * object_segmaps).sum(
#             dim=2
#         ) / object_segmaps_sum

#         return background_loss - object_loss

# def get_object_localization_loss_for_one_layer(
#     cross_attention_scores,
#     object_segmaps,
#     object_token_idx,
#     object_token_idx_mask,
#     loss_fn,
# ):
#     bxh, num_noise_latents, num_text_tokens = cross_attention_scores.shape
#     b, max_num_objects, _, _ = object_segmaps.shape
#     size = int(num_noise_latents**0.5)

#     # Resize the object segmentation maps to the size of the cross attention scores
#     object_segmaps = F.interpolate(
#         object_segmaps, size=(size, size), mode="bilinear", antialias=True
#     )  # (b, max_num_objects, size, size)

#     object_segmaps = object_segmaps.view(
#         b, max_num_objects, -1
#     )  # (b, max_num_objects, num_noise_latents)

#     num_heads = bxh // b

#     cross_attention_scores = cross_attention_scores.view(
#         b, num_heads, num_noise_latents, num_text_tokens
#     )

#     # Gather object_token_attn_prob
#     object_token_attn_prob = torch.gather(
#         cross_attention_scores,
#         dim=3,
#         index=object_token_idx.view(b, 1, 1, max_num_objects).expand(
#             b, num_heads, num_noise_latents, max_num_objects
#         ),
#     )  # (b, num_heads, num_noise_latents, max_num_objects)

#     object_segmaps = (
#         object_segmaps.permute(0, 2, 1)
#         .unsqueeze(1)
#         .expand(b, num_heads, num_noise_latents, max_num_objects)
#     )

#     loss = loss_fn(object_token_attn_prob, object_segmaps)

#     loss = loss * object_token_idx_mask.view(b, 1, max_num_objects)
#     object_token_cnt = object_token_idx_mask.sum(dim=1).view(b, 1) + 1e-5
#     loss = (loss.sum(dim=2) / object_token_cnt).mean()

#     return loss


# def get_object_localization_loss(
#     cross_attention_scores,
#     object_segmaps,
#     image_token_idx,
#     image_token_idx_mask,
#     loss_fn,
# ):
#     num_layers = len(cross_attention_scores)
#     loss = 0
#     for k, v in cross_attention_scores.items():
#         layer_loss = get_object_localization_loss_for_one_layer(
#             v, object_segmaps, image_token_idx, image_token_idx_mask, loss_fn
#         )
#         loss += layer_loss
#     return loss / num_layers


# clip_image_processor = CLIPImageProcessor()
# transform = transforms.Compose([
#             transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.CenterCrop(512),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
#         ])

# image_root_path="/data/zhangxujie/zwq/models/stable-diffusion-v1-5"
# json_file="/data/zhangxujie/zwq/IP-Adapter-main/file.json"
# file=open(json_file, 'r', encoding='utf-8')
# data=[]

# for line in file.readlines():
#     dic = json.loads(line)
#     data.append(dic)
#     segments=dic['segments']
#     # print(dic['segments'])
#     image_path = image_root_path + "/img/id_" + dic['image_id'] + ".jpg"
#     # print(image_path)
#     # print(os.path.exists(image_path))
#     # raw_image = Image.open(image_path)
#     # image = transform(raw_image.convert("RGB"))
#     # clip_image = clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
#     # print(clip_image)

#     # for seg in segments:
#     #     print(seg['id'])

#     for curDir, dirs, files in os.walk(image_root_path + "/seg/id_" + dic['image_id']):
#         for file in files:
#             if file.split("_")[2]
# image_encoder = CLIPVisionModelWithProjection.from_pretrained("/data/zhangxujie/zwq/models/CLIP-ViT-H-14")
# vision_model = image_encoder.vision_model
# visual_projection = image_encoder.visual_projection
# print(vision_model.config.image_size)
# text_encoder = CLIPTextModel.from_pretrained("/data/zhangxujie/zwq/models/stable-diffusion-v1-5/text_encoder")
# # text_encoder.to(accelerator.device, dtype=weight_dtype)
# tokenizer = CLIPTokenizer.from_pretrained("/data/zhangxujie/zwq/models/stable-diffusion-v1-5/tokenizer")
# embed_dim = text_encoder.config.hidden_size

# dic={"image_id": "000000001", "caption": "body piece,sleeve,collar,sleeve,top,", "segments": [{"id": "4", "start": 0, "end": 10, "bbox": [0, 0, 512, 512], "coco_label": "body piece"}, {"id": "1", "start": 11, "end": 17, "bbox": [423, 68, 510, 267], "coco_label": "sleeve"}, {"id": "3", "start": 18, "end": 24, "bbox": [188, 10, 324, 71], "coco_label": "collar"}, {"id": "2", "start": 25, "end": 31, "bbox": [3, 67, 89, 269], "coco_label": "sleeve"}, {"id": "0", "start": 32, "end": 35, "bbox": [0, 0, 512, 512], "coco_label": "top"}]}
# image_id=dic["image_id"]
# segments=dic["segments"]

# def read_mask(id,label):
#     if label=="body piece":
#         mask_path=mask_path="/data/zhangxujie/dataset/farfetch512/train_big/seg/id_"+ image_id +"/id_" + image_id + "_" + id + "_body_piece" + ".bmp"
#     else:    
#         mask_path=mask_path="/data/zhangxujie/dataset/farfetch512/train_big/seg/id_"+ image_id +"/id_" + image_id + "_" + id + "_" + label + ".bmp"
#     mask=cv2.imread(mask_path)[...,0:1]
#     return mask

# def get_part_clip_image(raw_image,mask):
#     Mask1=np.empty((512,512,1)).astype(np.uint8)
#     Mask1[mask >=255] = 1
#     Mask1[mask <255] = 0 
#     part=raw_image*Mask1
#     part_clip_image = clip_image_processor(images=part, return_tensors="pt").pixel_values
#     # print(type(part_clip_image))
#     # cv2.imwrite("1.jpg", part)
#     return part_clip_image

# raw_image = Image.open("/data/zhangxujie/dataset/farfetch512/train_big/img/id_000000001.jpg")
# image = transform(raw_image.convert("RGB"))
# clip_image = clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
# # mask1=cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/seg/id_000000001/id_000000001_1_sleeve.bmp")[...,0:1]
# # mask2=cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/seg/id_000000001/id_000000001_2_sleeve.bmp")[...,0:1]
# # mask=[mask1,mask2]
# mask=[read_mask(seg["id"],seg["coco_label"]) for seg in segments]

# text_origin="body piece,sleeve,collar,sleeve,top,"
# text=text_origin.replace("body piece","body")
# print(text)
# text_input_ids=tokenizer(
#             text,
#             max_length=tokenizer.model_max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors="pt"
#         ).input_ids
# encoder_hidden_states = text_encoder(text_input_ids)
# print(tokenizer.model_max_length) #[12,768]

# max_num_objects=text_input_ids.shape[1] #11
# num_objects=len(mask)
# # print(max_num_objects)

# part_clip_image=[get_part_clip_image(raw_image,item) for item in mask]
# print(part_clip_image[0])
# object_pixel_values = []
# padding_object_pixel_values = torch.zeros_like(part_clip_image[0])
# id_list=[0]*max_num_objects
# for seg in segments:
#     end=seg["end"]
#     list_short=text_origin[0:end+1]
#     n=2*list_short.count(",")-1
#     # print(list_short)
#     id_list[n]=1

# # print(id_list)
# num=0
# for id in id_list:
#     if id==0:
#         object_pixel_values+= [torch.zeros_like(padding_object_pixel_values)]
#     else:
#         object_pixel_values.append(part_clip_image[num])
#         num=num+1
# # print(object_pixel_values)

# # object_pixel_values = torch.tensor( [item.cpu().detach().numpy() for item in object_pixel_values] )
# # object_pixel_values=torch.Tensor(object_pixel_values)
# part_embeds = image_encoder(object_pixel_values[0]).image_embeds
# print("11111",part_embeds.shape)

# part_embeds=[image_encoder(item).image_embeds for item in object_pixel_values]
# part_embeds=torch.cat([item for item in part_embeds],dim=0)

# mlp1 = MLP(embed_dim + 1024, embed_dim , embed_dim, use_residual=False)
# mlp2 = MLP(embed_dim , embed_dim, embed_dim, use_residual=True)
# layer_norm = nn.LayerNorm(embed_dim)

# black=image_encoder(torch.zeros_like(padding_object_pixel_values)).image_embeds
# part_embeds=[]
# n=0
# for object_pixel_values, id_ in zip(object_pixel_values,id_list):
#     # print(object_pixel_values.shape,id_)
#     if id_==1:
#         part_embeds = image_encoder(object_pixel_values).image_embeds
#         part_embeds = part_embeds.view(-1)
#         # print(part_embeds.shape)
#         text_object_embeds = torch.cat([encoder_hidden_states[0][0][n], part_embeds], dim=-1)
#         # print("1:",text_object_embeds.shape)
#         text_object_embeds = mlp1(text_object_embeds) + encoder_hidden_states[0][0][n]
#         text_object_embeds = mlp2(text_object_embeds)
#         text_object_embeds = layer_norm(text_object_embeds)
#         encoder_hidden_states[0][0][n] = text_object_embeds
        
# #         # print("1")
# # # print(len(part_embeds))
# print("2:",encoder_hidden_states[0][0].shape)



# part_embeds=torch.cat([item for item in part_embeds],dim=0)
# part_embeds=torch.unsqueeze(part_embeds, dim=0)
# print("part:",part_embeds.shape)#[1 77 1024]
# text_object_embeds = torch.cat([encoder_hidden_states[0][0], part_embeds], dim=-1)#[11,2048]
# print("1:",mlp1(text_object_embeds[0]).shape)
# text_object_embeds = mlp1(text_object_embeds) + encoder_hidden_states[0][0]
# print("2:",mlp1(text_object_embeds).shape)
# text_object_embeds = mlp2(text_object_embeds)
# text_object_embeds = layer_norm(text_object_embeds)#[11,1024]
# # text_object_embeds = torch.unsqueeze(text_object_embeds, dim=0)



# print("embed_dim:",embed_dim)
# print(text)
# print(text_input_ids.shape)
# print(encoder_hidden_states[0][0].shape)#[77,768]
# # print(part_embeds)
# # print(part_embeds.shape)
# print("text_objext:",text_object_embeds.shape)




# class ImageProjModel(torch.nn.Module):
#     """Projection Model"""

#     def __init__(self,cross_attention_dim=768, clip_embeddings_dim=768, clip_extra_context_tokens=1):
#         super().__init__()

#         self.cross_attention_dim = cross_attention_dim
#         self.clip_extra_context_tokens = clip_extra_context_tokens
#         self.proj = torch.nn.Linear(768, 1 * 768)
#         self.norm = torch.nn.LayerNorm(768)

#     def forward(self, image_embeds):
#         embeds = image_embeds
#         clip_extra_context_tokens = self.proj(embeds).reshape(
#             -1, self.cross_attention_dim
#         )
#         clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
#         return clip_extra_context_tokens

# image_proj_model = ImageProjModel()
# ip_tokens = image_proj_model(encoder_hidden_states[0][0])
# print(ip_tokens.shape)

# proj = torch.nn.Linear(768, 1 * 768)
# norm=torch.nn.LayerNorm(768)
# print(proj(encoder_hidden_states[0][0]).shape)
# hehe = proj(encoder_hidden_states[0][0]).reshape(-1, 768)
# hehe = norm(hehe)
# print(hehe.shape)



# unet = UNet2DConditionModel.from_pretrained(
#             "/data/zhangxujie/zwq/models/stable-diffusion-v1-5/unet"
#         )

# def unet_store_cross_attention_scores(unet, attention_scores, layers=5):
#     from diffusers.models.attention_processor import (
#         Attention,
#         AttnProcessor,
#         AttnProcessor2_0,
#     )

#     UNET_LAYER_NAMES = [
#         "down_blocks.0",
#         "down_blocks.1",
#         "down_blocks.2",
#         "mid_block",
#         "up_blocks.1",
#         "up_blocks.2",
#         "up_blocks.3",
#     ]

#     start_layer = (len(UNET_LAYER_NAMES) - layers) // 2
#     end_layer = start_layer + layers
#     applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

#     def make_new_get_attention_scores_fn(name):
#         def new_get_attention_scores(module, query, key, attention_mask=None):
#             attention_probs = module.old_get_attention_scores(
#                 query, key, attention_mask
#             )
#             attention_scores[name] = attention_probs
#             return attention_probs

#         return new_get_attention_scores

#     for name, module in unet.named_modules():
#         if isinstance(module, Attention) and "attn2" in name:
#             if not any(layer in name for layer in applicable_layers):
#                 continue
#             if isinstance(module.processor, AttnProcessor2_0):
#                 module.set_processor(AttnProcessor())
#             module.old_get_attention_scores = module.get_attention_scores
#             module.get_attention_scores = types.MethodType(
#                 make_new_get_attention_scores_fn(name), module
#             )

#     return unet
# # cross_attention_scores = {}
# # unet = unet_store_cross_attention_scores(
# #                 unet, cross_attention_scores, localization_layers
# #             )

# def get_object_processor(args):
#     if args.object_background_processor == "random":
#         object_processor = RandomSegmentProcessor()
#     else:
#         raise ValueError(f"Unknown object processor: {args.object_processor}")
#     return object_processor

# class SegmentProcessor(torch.nn.Module):
#     def forward(self, image, background, segmap, id, bbox):
#         # mask = segmap != id
#         mask=segmap
#         print("111",mask.shape)
#         image[mask, :] = background[mask, :]
#         h1, w1, h2, w2 = bbox
#         return image[w1:w2, h1:h2,:]

#     def get_background(self, image):
#         raise NotImplementedError
    
# class RandomSegmentProcessor(SegmentProcessor):
#     def get_background(self, image):
#         background = torch.randint(
#             0, 255, image.shape, dtype=torch.float32
#         )#+device
#         return background
    
# from copy import deepcopy  
# from torchvision.io import read_image, ImageReadMode  
# # print(raw_image.shape)
# bbox = [3, 67, 89, 269]
# img111=torch.tensor(np.asarray(raw_image)).float()
# print("img",img111.shape)
# # segmap=torch.tensor(np.expand_dims(mask111.squeeze(),2).repeat(3,axis=2))
# segmap=torch.tensor(mask111.squeeze())
# print("mask:",segmap.shape)
# object_processor = RandomSegmentProcessor()
# background = object_processor.get_background(img111)
# print("background",background.shape)
# object_image = object_processor(
#     deepcopy(img111), background, segmap, id, bbox
# )
# print(object_image.shape)
# pic = object_image.detach().cpu().numpy()
# cv2.imwrite("1111.jpg",pic)
# cv2.imwrite("bg.jpg",background.detach().cpu().numpy())


# for curDir, dirs, files in os.walk("/data/zhangxujie/zwq/datasets/mscoco"):
#     print(curDir)
#     for curDir_, dirs_, files_ in os.walk(curDir):
#         for file in files_:
#             if file.endswith(".jpg"):
#                 print(os.path.join(curDir_,file))
#                 print(os.path.join(curDir_,file).replace("jpg","txt"))
                