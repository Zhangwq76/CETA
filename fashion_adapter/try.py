# from transformers import CLIPImageProcessor
# from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
# from torchvision import transforms
# from PIL import Image
# from accelerate import Accelerator
# import torch.nn as nn
# import torch
# from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel

# image_encoder = CLIPVisionModelWithProjection.from_pretrained("/data/zhangxujie/zwq/models/CLIP-ViT-H-14")
# text_encoder = CLIPTextModel.from_pretrained("/data/zhangxujie/zwq/models/stable-diffusion-v1-5/text_encoder")
# tokenizer = CLIPTokenizer.from_pretrained("/data/zhangxujie/zwq/models/stable-diffusion-v1-5/tokenizer")

# clip_image_processor = CLIPImageProcessor()

# raw_image = Image.open("/data/zhangxujie/dataset/farfetch512/train_big/img/id_000000001.jpg")
# clip_image = clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
# clip_image2 = torch.randn(1,3,224,224)
# clip=[clip_image,clip_image2]
# clip=torch.cat([img for img in clip], dim=0) #[2,3,224,224]
# input_k = image_encoder(clip).image_embeds
# # input_k = input_k.view(1,1,-1)
# # input_k = torch.randn(1, 1, 1024)
# # input_v = input_k
# print(input_k.shape)

# text='sleeve,sleeve,body piece,dress,'
# text=text[0:-1]
# text = text.split(',')
# text1 = text.pop(3)
# print(text1,text)

# from einops import rearrange
# text_input_ids=tokenizer(
#             text,
#             max_length=4,
#             padding='max_length',
#             truncation=True,
#             return_tensors="pt"
#         ).input_ids
# print(text_input_ids.shape)
# text_input_ids = torch.stack([text_input_ids,text_input_ids]) #[2,4,4]
# bs, _, _=text_input_ids.shape
# print("111",text_input_ids.shape)
# input_ids = rearrange(text_input_ids, 'b n t -> (b n) t') #[8,4]
# print("222",input_ids.shape)
# encoder_hidden_states = text_encoder(input_ids)[0] #[8,4,768]
# print("333",encoder_hidden_states.shape)
# encoder_hidden_states = rearrange(encoder_hidden_states, '(b n) t d -> b n t d', b=bs)
# print("444", encoder_hidden_states.shape)
# input_q = encoder_hidden_states[0] #[768]
# # input_q = input_q.view(1,1,-1)
# # clip_img=image_encoder(clip).image_embeds
# print(input_q.shape)



# class CrossAttention(nn.Module):
#     def __init__(self, dim, heads = 8):
#         super().__init__()
#         self.dim = dim
#         self.heads = heads
#         self.scale = dim ** -0.5
#         self.query = nn.Linear(1024, dim)
#         self.key = nn.Linear(dim, dim)
#         self.value = nn.Linear(dim, dim)
        
#     def forward(self, queries, keys, values, mask = None):
#         b, n, _, h = *queries.shape, self.heads
#         b_k, n_k, _, h = *keys.shape, self.heads
#         queries = self.query(queries)
#         keys = self.key(keys)
#         values = self.value(values)
#         queries = queries.view(b, n, h, -1).transpose(1, 2)
#         print(queries.shape)
#         keys = keys.view(b_k, n_k, h, -1).transpose(1, 2)
#         values = values.view(b, n, h, -1).transpose(1, 2)
#         print("queries:",queries.shape,"keys:",keys.shape)
        
#         dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale
#         print(dots.shape)
        
#         if mask is not None:
#             mask = F.pad(mask.flatten(1), (1, 0), value = True)
#             assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
#             mask = mask[:, None, :].expand(-1, n, -1)
#             dots.masked_fill_(~mask, float('-inf'))
            
#         attn = dots.softmax(dim=-1)
#         out = torch.einsum('bhij,bhjd->bhid', attn, values)
#         out = out.transpose(1, 2).contiguous().view(b_k, n_k, -1)
#         return out
    
# input_q=torch.randn(8,1,1024)
# input_k=torch.randn(8,4,1024)
# print(torch.unsqueeze(input_k,dim=0).shape)
# # dots = torch.einsum('bhid,bhjd->bhij', input_q, input_k)
# # print(dots.shape)
# input_v=input_k

# cross_attention = CrossAttention(1024)
# out=cross_attention(input_q, input_k,input_v)
# print(out.shape)

# from safetensors import safe_open
# import torch
# tensors = {}
# opt = torch.load("/data/zhangxujie/zwq/IP-Adapter-main/fb_18500_16.bin", map_location="cpu")
# for k in opt:
#     print(k)
# with safe_open("sd-ip_adapter/checkpoint-170000/model.safetensors", framework="pt", device=0) as f:
#     for k in f.keys():
#         print(k)

# import cv2
# import numpy as np
# image = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/img/id_000004668.jpg").astype(np.uint8)
# # mask_body = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/seg/id_000000083/id_000000083_18_body_piece.bmp")
# # mask_body = (mask_body==255).astype(np.uint8)
# # mask_sleeve = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/seg/id_000000083/id_000000083_1_sleeve.bmp")
# # mask_sleeve = (mask_sleeve==255).astype(np.uint8)
# mask_collar = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/seg/id_000004668/id_000004668_4_collar.bmp")
# mask_collar = (mask_collar==255).astype(np.uint8)


# # image1 = image*mask_body
# # image2 = image*mask_sleeve
# image3 = image*mask_collar

# # cv2.imwrite("test_img/83_18_body.jpg",image1)
# # cv2.imwrite("test_img/83_1_sleeve.jpg",image2)
# cv2.imwrite("test_img/4668_4_collar.jpg",image3)

from PIL import Image
file=open("/data/zhangxujie/dataset/farfetch512/train_big/train_big_new.txt", 'r', encoding='utf-8')
data_=[]
for line in file.readlines():  
    data_.append(line)
data = data_

for line in data:   
    image_root_path = "/data/zhangxujie/dataset/farfetch512/train_big"

    line = line[0:-1]
    line = line.split(" ")
    image_id=line[0]
    text_id=line[1]
    print(image_id)
    print(text_id)

    # image_path = image_root_path + "/img/" + image_id
    # text_path = image_root_path + "/txt/" + text_id

    # # read image
    # raw_image = Image.open(image_path)
    # if raw_image is not None:
    #     print(image_path)