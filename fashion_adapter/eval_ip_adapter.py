import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image

from ip_adapter import IPAdapter

base_model_path = "/data/zhangxujie/zsy/A-Model/fashion-sd-2.1"
vae_model_path = "/data/zhangxujie/zsy/A-Model/fashion-sd-2.1/vae"
image_encoder_path = "/data/zhangxujie/zwq/models/CLIP-ViT-H-14"
ip_ckpt = "/data/zhangxujie/zwq/IP-Adapter-main/fashion_adapter_260000.bin"
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

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

# load SD Img2Img pipe
torch.cuda.empty_cache()
pipe =  StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# read image prompt
image = Image.open("/data/zhangxujie/zwq/IP-Adapter-main/test_img/band1.png")
g_image = Image.open("/data/zhangxujie/zwq/IP-Adapter-main/test_img/band1.png")

ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

images = ip_model.generate(pil_image=image, num_samples=8, num_inference_steps=50, seed=2004466, guidance_scale=7.5,scale=0.3,
                           prompt="A skirt,high quality")

grid = image_grid(images, 2, 4)

grid.save("eval/testtt.png")