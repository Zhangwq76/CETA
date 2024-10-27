import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import cv2
import numpy as np
import torch.nn as nn

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        # self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]
        file=open(json_file, 'r', encoding='utf-8')
        data_=[]
        for line in file.readlines():
            dic = json.loads(line)
            data_.append(dic)
        self.data=data_

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()


        
    def __getitem__(self, idx):
        
        def read_mask(id,label):
            if label=="body piece":
                mask_path=mask_path="/data/zhangxujie/dataset/farfetch512/train_big/seg/id_"+ image_id +"/id_" + image_id + "_" + id + "_body_piece" + ".bmp"
            elif label=="band":
                mask_path=mask_path="/data/zhangxujie/dataset/farfetch512/train_big/seg/id_"+ image_id +"/id_" + image_id + "_" + id + "_waist_band" + ".bmp"
            else:    
                mask_path=mask_path="/data/zhangxujie/dataset/farfetch512/train_big/seg/id_"+ image_id +"/id_" + image_id + "_" + id + "_" + label + ".bmp"
            if os.path.exists(mask_path)==False:
                print(mask_path)
            mask=cv2.imread(mask_path)[...,0:1]
            return mask

        def get_part_clip_image(raw_image,mask):
            Mask1=np.empty((512,512,1)).astype(np.uint8)
            Mask1[mask >=255] = 1
            Mask1[mask <255] = 0 
            part=raw_image*Mask1
            part_clip_image = self.clip_image_processor(images=part, return_tensors="pt").pixel_values
            # part_clip_image=vision_processor(torch.Tensor(part)).numpy()
            # cv2.imwrite("1.jpg", part)
            return part_clip_image
        
        item = self.data[idx] 
        # text = item["text"]
        # image_file = item["image_file"]
        
        text_origin = item['caption']
        text=text_origin.replace("body piece","body")
        text = text.replace(","," ")
        image_id=item['image_id']
        segments=item['segments']

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding='longest',
            truncation=True,
            return_tensors="pt"
        ).input_ids

        max_num_objects=text_input_ids.shape[1]

        if max_num_objects>2:
            # encoder_hidden_states = self.text_encoder(text_input_ids)
            image_path = self.image_root_path + "/img/id_" + image_id + ".jpg"

            # read image
            # raw_image = Image.open(os.path.join(self.image_root_path, image_file))
            # image = self.transform(raw_image.convert("RGB"))
            # clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
            raw_image = Image.open(image_path)
            image = self.transform(raw_image.convert("RGB"))
            clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
            mask=[read_mask(seg["id"],seg["coco_label"]) for seg in segments]
            part_clip_image=[get_part_clip_image(raw_image,item) for item in mask]
            padding_object_pixel_values = torch.zeros_like(part_clip_image[0])

            # drop
            drop_image_embed = 0
            rand_num = random.random()
            if rand_num < self.i_drop_rate:
                drop_image_embed = 1
            elif rand_num < (self.i_drop_rate + self.t_drop_rate):
                text = ""
            elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
                text = ""
                drop_image_embed = 1


            # id_list=[0]*max_num_objects
            # # print(max_num_objects)
            # for seg in segments:
            #     end=seg["end"]
            #     list_short=text_origin[0:end+1]
            #     n=2*list_short.count(",")-1
            #     # print(n)
            #     id_list[n]=1

            object_pixel_values = torch.cat([img for img in part_clip_image],dim=0)
            # num=0
            # for id in id_list:
            #     if id==0:
            #         object_pixel_values+= [torch.zeros_like(padding_object_pixel_values)]
            #     else:
            #         object_pixel_values.append(part_clip_image[num])
            #         num=num+1

            return {
                "image": image,
                "text_input_ids": text_input_ids,
                "clip_image": clip_image,
                "drop_image_embed": drop_image_embed,
                "part_clip_image":part_clip_image,
                "object_pixel_values":object_pixel_values
            }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids_ = [example["text_input_ids"] for example in data]
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    part_clip_image = [example["part_clip_image"] for example in data]
    object_pixel_values = [example["object_pixel_values"] for example in data]

    return {
        "images": images,
        "text_input_ids_": text_input_ids_,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "part_clip_image":part_clip_image,
        "object_pixel_values":object_pixel_values
    }
    
class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        self.query = nn.Linear(768, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, queries, keys, values, mask = None):
        # print("queries:",queries.shape)
        b, n, _, h = *queries.shape, self.heads
        b_k, n_k, _, h = *keys.shape, self.heads
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)
        queries = queries.view(b, n, h, -1).transpose(1, 2)
        keys = keys.view(b_k, n_k, h, -1).transpose(1, 2)
        values = values.view(b, n, h, -1).transpose(1, 2)
        # print("queries:",queries.shape,"keys:",keys.shape)
        
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale
        
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, n, -1)
            dots.masked_fill_(~mask, float('-inf'))
            
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        out = out.transpose(1, 2).contiguous().view(b_k, n_k, -1)
        return out

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, crossattention, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.crossattention = crossattention

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds, text_embeds):
        # ip_tokens = torch.Tensor()
        # for n in range(image_embeds.shape[0]): #[8,7,1024]
        #     image_embed = image_embeds[n] #[7,1024]
        #     ip_tokens_ = torch.Tensor()
        #     for m in range(image_embed.shape[0]):
        #         part_embed = image_embed[m] #[1024]
        #         if not torch.equal(part_embed, torch.zeros_like(part_embed)): 
        #             ip_token = self.image_proj_model(part_embed.view(1,1,-1)) #[1,1,768]
        #             # print("IP:",ip_token.shape)
        #             IP_token = self.CrossAttention(text_embeds[n][m].view(1,1,-1), ip_token, ip_token) #[1,1,768]
        #             ip_tokens_ = torch.cat([ip_tokens_, IP_token], dim=0)
        #         else:
        #             ip_tokens_ = torch.cat([ip_tokens_, IP_token], dim=0) #[77,1,768]
            # ip_tokens = torch.cat([ip_tokens,ip_tokens_], dim=1)
        # ip_tokens = ip_tokens.transpose(0,1) #[8,77,768]
        # print("333",ip_tokens.shape,encoder_hidden_states.shape)
        
        ip_token = self.image_proj_model(image_embeds) #[8,7,1024]
        ip_tokens = self.crossattention(text_embeds, ip_token, ip_token) #[8,7,768]
        
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data/zhangxujie/zwq/models/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default="/data/zhangxujie/zwq/IP-Adapter-main/data.json",
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
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
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
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
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
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # embed_dim = text_encoder.config.hidden_size
    # print("embed_dim:",embed_dim)
    # mlp1 = MLP(embed_dim + 1024, embed_dim, embed_dim, use_residual=False).to(accelerator.device)
    # mlp2 = MLP(embed_dim , embed_dim, embed_dim, use_residual=True).to(accelerator.device)
    # layer_norm = nn.LayerNorm(embed_dim).to(accelerator.device)

    #ip-adapter
    print("1111",unet.config.cross_attention_dim)
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=1,
    )
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
    crossAttention = CrossAttention(768)
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, crossAttention, args.pretrained_ip_adapter_path)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0] 
                    # print("222",encoder_hidden_states.shape)  
                    encoder_hidden_states_ = [text_encoder(text_input_ids_.to(accelerator.device, dtype=torch.long))[0] for text_input_ids_ in batch["text_input_ids_"]]
                    # encoder_hidden_states = [item[0] for item in batch["encoder_hidden_states"]]
                    # encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]  

                with torch.no_grad():
                    image_embeds_ = []
                    text_embeds_ = []
                    max_num = 7
                    for object_pixel_values,encoder_hidden_states_ in zip(batch["object_pixel_values"],encoder_hidden_states_):
                        part_embeds_=torch.Tensor()
                        word_embeds_=torch.Tensor()
                        
                        part_embeds_ = torch.cat([part_embeds_, torch.zeros(1,1024)])
                        part_embeds = image_encoder(object_pixel_values.to(accelerator.device,dtype=weight_dtype)).image_embeds #[5,1024]
                        part_embeds_ = torch.cat([part_embeds_, part_embeds],dim=0)
                        word_embeds_ = encoder_hidden_states[0] #[7,768]
                                   
                        if len(part_embeds_)<max_num:
                            part_embeds_ = torch.cat([part_embeds_, torch.zeros(max_num-part_embeds_.shape[0],1024)],dim=0) #[7,1024]
                            
                        image_embeds_.append(part_embeds_)
                        text_embeds_.append(word_embeds_)
                    image_embeds = torch.stack(image_embeds_)
                    # print(image_embeds.shape)
                    text_embeds = torch.stack(text_embeds_)
                    # print("image_embeds",image_embeds.shape)
                ##drop
                # image_embeds_ = []
                # for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                #     if drop_image_embed == 1:
                #         image_embeds_.append(torch.zeros_like(image_embed))
                #     else:
                #         image_embeds_.append(image_embed)
                # image_embeds = torch.stack(image_embeds_)
                
                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds,text_embeds)
        
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") #denoise_loss
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
