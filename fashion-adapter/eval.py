import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import cv2
import numpy as np
# from ip_adapter import IPAdapter
import itertools
import os

from transformers import CLIPImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
import torch.nn as nn
import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, CNAttnProcessor2_0 as CNAttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor, CNAttnProcessor
    

image_encoder = CLIPVisionModelWithProjection.from_pretrained("/data/zhangxujie/zwq/models/CLIP-ViT-H-14")
text_encoder = CLIPTextModel.from_pretrained("/data/zhangxujie/zsy/A-Model/fashion-sd-2.1/text_encoder")
tokenizer = CLIPTokenizer.from_pretrained("/data/zhangxujie/zsy/A-Model/fashion-sd-2.1/tokenizer")
base_model_path = "/data/zhangxujie/zsy/A-Model/fashion-sd-2.1"
vae_model_path = "/data/zhangxujie/zsy/A-Model/fashion-sd-2.1/vae"
# base_model_path = "/data/zhangxujie/zwq/models/stable-diffusion-v1-5"
# vae_model_path = "/data/zhangxujie/zwq/models/stable-diffusion-v1-5/vae"
clip_image_processor = CLIPImageProcessor()

fb_ckpt = "/data/zhangxujie/zwq/IP-Adapter-main/fashion_fb_27500_44w.bin" #fb
# ip_ckpt = "/data/zhangxujie/zwq/models/ip-adapter_sd15.bin"
ip_ckpt = "/data/zhangxujie/zwq/IP-Adapter-main/fashion_adapter_260000.bin"
device = "cuda"

image_token_num=16

# load ip-adapter
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
).to(device)

vae.to(device, dtype=torch.float16)
text_encoder.to(device, dtype=torch.float16)
image_encoder.to(device, dtype=torch.float16)

class ImageProjModel0(torch.nn.Module):
    """Projection Model for IP-adapter"""

    def __init__(self, cross_attention_dim=pipe.unet.config.cross_attention_dim, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class ImageProjModel(torch.nn.Module):
    """Projection Model for feature_blender"""

    def __init__(self, cross_attention_dim=pipe.unet.config.cross_attention_dim, clip_embeddings_dim=1024, clip_extra_context_tokens=image_token_num):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(1024, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(self.cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        bs, num, _ = embeds.shape
        # print("embeds:",embeds.shape)
        clip_extra_context_tokens = self.proj(embeds).reshape(
            bs, num, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class FeatureBlender(nn.Module):
    def __init__(self,
                 num_output_tokens=1,
                 num_text_tokens=4,
                 num_objects=4, 
                 text_token_dim=text_encoder.config.hidden_size,
                 image_token_dim=pipe.unet.config.cross_attention_dim,
                 cross_attention_dim=pipe.unet.config.cross_attention_dim,
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
        queries = self.to_query(text_embeds)  # [batch_size, num_objects, num_text_tokens, cross_attention_dim]
        keys = self.to_key(image_features)  # [batch_size, num_objects, num_text_tokens, cross_attention_dim]
        values = self.to_value(image_features)  # [batch_size, num_objects, num_text_tokens, cross_attention_dim]

        # FIXME: 并行处理的原理是先把 部件维度 和 batch 维度合并，然后再拆分
        # To [batch_size * num_objects, num_tokens, num_heads, cross_attention_dim // num_heads]
        queries, keys, values = map(lambda x: rearrange(x, 'b n t (h d) -> (b n) t h d', h=self.heads).transpose(1, 2),
                                    (queries, keys, values))

        # Need PyTorch 2.0+
        assert is_torch2_available(), "This module requires PyTorch 2.0+"
        hidden_states = F.scaled_dot_product_attention(
            queries, keys, values, dropout_p=self.dropout, is_causal=False
        )
        hidden_states = rearrange(hidden_states, '(b n) t h d -> b n t (h d)', b=b)
        # To [batch_size, num_output_tokens, cross_attention_dim]
        hidden_states = hidden_states.view(b, self.num_output_tokens, -1)
        hidden_states = hidden_states.to(queries.dtype)
        # print(hidden_states.shape)
        
        # To out MLP
        out = self.to_out(hidden_states)  # [batch_size, num_output_tokens, text_token_dim][1,4,1024]
        print("out:",out.shape)
        return out

class IPAdapter:
    def __init__(self, 
                 sd_pipe=None, 
                 image_encoder=None, 
                 ip_ckpt="/data/zhangxujie/zwq/IP-Adapter-main/ip_adapter_1024_30000new.bin", 
                 device="cuda", 
                 num_tokens=4, 
                 Tokenizer=None, 
                 text_encoder=None, 
                 feature_blender=None,
                 fb_proj_model=None):
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()
        self.tokenizer = Tokenizer
        self.text_encoder = text_encoder
        self.feature_blender = feature_blender
        self.fb_proj_model = fb_proj_model
        
        # load image encoder
        self.image_encoder = image_encoder
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel0(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
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
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=8,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        raw_image = cv2.imread("/data/zhangxujie/zwq/IP-Adapter-main/test_img/5.png") #1-2
        raw_image1 = cv2.imread("/data/zhangxujie/zwq/IP-Adapter-main/test_img/267652_hood.png")
        # raw_image1 = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/img/id_000000001.jpg")#47-3
        raw_image2 = cv2.imread("/data/zhangxujie/zwq/IP-Adapter-main/test_img/30084_sleeve.jpg") #83_5
        raw_image3 = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/img/id_000015245.jpg")#97-6
        # part1_mask = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/seg/id_000000047/id_000000047_3_collar.bmp")
        # part1_mask = (part1_mask==255).astype(np.uint8)
        # garment=raw_image*part1_mask
        # cv2.imwrite("collar.jpg",garment)
        mask = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/seg/id_000000084/id_000000084_1_waist_band.bmp")
        mask = (mask==255).astype(np.uint8)
        mask1 = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/seg/id_000030181/id_000030181_3_collar.bmp")
        mask1 = (mask1==255).astype(np.uint8)
        mask2 = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/seg/id_000000028/id_000000028_4_pocket.bmp")
        mask2 = (mask2==255).astype(np.uint8)
        mask3 = cv2.imread("/data/zhangxujie/dataset/farfetch512/train_big/seg/id_000015245/id_000015245_6_body_piece.bmp")
        mask3 = (mask3==255).astype(np.uint8)

        image = raw_image
        image1 = raw_image1
        image2 = raw_image2
        image3 = raw_image3*mask3
        
        # empty = np.empty([512,512,3], dtype=float).astype(np.uint8)
        
        # image = raw_image
        # image1 = raw_image1
        # image2 = raw_image1
        # image3 = raw_image3

        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        image1 = Image.fromarray(cv2.cvtColor(image1,cv2.COLOR_BGR2RGB))
        image2 = Image.fromarray(cv2.cvtColor(image2,cv2.COLOR_BGR2RGB))
        image3 = Image.fromarray(cv2.cvtColor(image3,cv2.COLOR_BGR2RGB))    
        
        # image.save("test_img/84_1_band.jpg")
        # image1.save("test_img/30181.jpg")
        # image2.save("test_img/16247_3_hood.jpg")
        # image3.save("test_img/15245_6_body.jpg")

        empty = torch.zeros(1, 3, 224, 224)
        clip_image = self.clip_image_processor(images=image, return_tensors="pt").pixel_values
        clip_image1 = self.clip_image_processor(images=image1, return_tensors="pt").pixel_values
        clip_image2 = self.clip_image_processor(images=image2, return_tensors="pt").pixel_values
        clip_image3 = self.clip_image_processor(images=image3, return_tensors="pt").pixel_values
        clip = torch.cat([clip_image,clip_image1,empty,empty],dim=0)
        
        # clip = self.clip_image_processor(images=image, return_tensors="pt").pixel_values
        # image_embeds = self.image_encoder(clip.to(device, dtype=torch.float16)).image_embeds
        
        image_embeds = self.image_encoder(clip.to(device, dtype=torch.float16)).image_embeds.unsqueeze(0) #[1,4,1024]
        text = ["body piece","hood","",""]
        # text = ["","","",""]
        text_input_ids=self.tokenizer(
                    text,
                    max_length=4,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
        # print(text_input_ids.shape)#[4,4]
        encoder_hidden_states = self.text_encoder(text_input_ids.to(device))[0] #[4,4,1024]
        text_feature = encoder_hidden_states.unsqueeze(0)
        image_features = self.fb_proj_model(image_embeds)#[1,4,16,1024]
        print(image_features.shape)
        part_feature = self.feature_blender(image_features,text_feature) #[1,1,1024]
        
        image_prompt_embeds = part_feature
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(clip_image_embeds = part_feature)
        print("part:",image_prompt_embeds.shape) #[1,4,768]
        # uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds))

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        # uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        # uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        prompt = "hoodie,high quality" #short sleeve T-shirt with collar and pocket
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        # print(prompt_embeds_.shape,negative_prompt_embeds_.shape)#[1,77,768]
        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, torch.zeros_like(image_prompt_embeds)], dim=1)
        # print(prompt_embeds.shape,negative_prompt_embeds.shape)

        seed=20466 #11111
        generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
            **kwargs,
        ).images
    
        return images

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


state_dict = torch.load(fb_ckpt, map_location="cpu")
feature_blender = FeatureBlender()
feature_blender.load_state_dict(state_dict["feature_blender"])
feature_blender.to(device, dtype=torch.float16)

fb_proj_model = ImageProjModel()
fb_proj_model.load_state_dict(state_dict["image_proj"])
fb_proj_model.to(device, dtype=torch.float16)

ip_adapter =  IPAdapter(pipe, image_encoder,ip_ckpt,device,4,tokenizer,text_encoder,feature_blender,fb_proj_model)
images=ip_adapter.generate()
grid = image_grid(images, 2, 4)
grid.save("eval/6.png")