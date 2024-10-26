# CETA：Fashion agent for Clothing Editing and Try-on Application

CETA is a fashion-oriented intelligent agent designed for editing, generating, and virtually trying on clothing items.

By utilizing advanced artificial intelligence models and text analysis technology, CETA can derive the required clothing suggestions from users' natural language input, generate customized clothing designs, and provide try-on display functions.

## Installation
Create a conda environment & Install requirments
```shell
conda create -n CETA python==3.9.0
conda activate CETA
pip install diffusers==0.22.1
pip install git+https://github.com/tencent-ailab/IP-Adapter.git
cd CETA/Catvton
pip install -r requirements.txt
```

## Download Models
To run CETA, you need to download the following models:
[CLIP-ViT-H-14](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
[Fahion-sd-2.1](https://huggingface.co/Zhangwq76/fashion-adapter/tree/main/fashion-sd-2.1)
[Catvton](https://huggingface.co/zhengchong/CatVTON)
[stable-diffusion-inpainting](https://huggingface.co/booksforcharlie/stable-diffusion-inpainting)

You can put these checkpoints in CETA/model folder.

## How to Use
You can simply run:
```shell
cd CETA/TelegramBotTest
python botTest.py
```
Then you can talk to the robot now!
