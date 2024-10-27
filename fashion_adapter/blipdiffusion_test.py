from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.utils import load_image
import torch

cond_subject = "sweater"
tgt_subject = "sweater"
text_prompt_input = "a sweater,high quality, reality"

cond_image = load_image(
    "/data/zhangxujie/zwq/IP-Adapter-main/test_img/bodypiece.jpg"
)
print(type(cond_image))

blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
    "/data/zhangxujie/zwq/models/Blipdiffusion", torch_dtype=torch.float16
).to("cuda")


guidance_scale = 7.5
num_inference_steps = 50
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


output = blip_diffusion_pipe(
    text_prompt_input,
    cond_image,
    cond_subject,
    tgt_subject,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    neg_prompt=negative_prompt,
    height=512,
    width=512,
).images
output[0].save("eval/blip_8.png")