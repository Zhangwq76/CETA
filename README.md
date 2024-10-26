# 🤖CETA：Fashion agent for Clothing Editing and Try-on Application

CETA is a fashion-oriented intelligent agent designed for editing, generating, and virtually trying on clothing items.

By utilizing advanced artificial intelligence models and text analysis technology, CETA can derive the required clothing suggestions from users' natural language input, generate customized clothing designs, and provide try-on display functions.

![image](https://github.com/user-attachments/assets/89bedefa-deda-454c-a5e3-b53d1663fec5)


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

Detailed user guide refers to appendix in final report

## Contributors
| Official Full Name  | Student ID (MTech Applicable) | Email |
| :------------ |:---------------:| :-----|
| Gong Xinyu | A0295988W | 1@u.nus.edu |
| Liu Peilin | A0295988W | 2@u.nus.edu |
| Su Chang | A0296250A | 3@u.nus.edu |
| Wang Xinji | A0265810H | 4@u.nus.edu |
| Zhang Wenqing | A0296886Y | 5@u.nus.edu |

