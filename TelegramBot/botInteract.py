from telegram import Update, InputFile
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from openai import OpenAI
import os
import sys
sys.path.append("E:/CodeAllNight/CETA/CETA/")  # sc改，改成项目根目录
from database.insert_images import insert_in_memory_table
from databaseAPI.get_recommendation import get_recommendation
from fashion_adapter.image_generation import generate_images  # 导入生成图像的函数
# sys.path.append("E:/CodeAllNight/CETA/CETA/tryon")
from try_on import tryon_process
# 状态跟踪字典
user_states = {}

# 定义 /start 命令的处理函数
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("你好！我是一个智能聊天机器人，向我发送任何问题，我将会帮你回答！")


# 定义与OpenAI ChatGPT进行对话的函数
def chat_with_gpt(message: str) -> str:
    try:
        # 使用聊天接口生成回复
        print("Sending message to OpenAI API...")
        client = OpenAI(
           api_key = 'sk-svcacct-WaBPRBqFS9ICIVNSIbKBoja6n5yfdtKtnkncf-xAh5koGdGeht7NsgSFWbRjKYmXpzvT3BlbkFJ4B4NorQxTMSdbjw4HKAaHpeKzv5t3Wgr_o8TA4N41Tc8tx2MkpG3Ah6o2ywvgULci7gA'
        )
        response  = client.chat.completions.create(
            messages=[
                 {"role": "system", "content": "You are a helpful cloth design assistant. If the user ask you something not relevant with the topic of clothes, you should kindly mind them that this bot is specifically for processing clothes."},  # 系统消息设定助手行为
                 {"role": "user", "content": message}  # 用户输入
            ],
            model="gpt-4o-mini",
)
        # 提取 ChatGPT 的回复
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"发生错误: {str(e)}"


# 处理 /search 命令的函数
def search(update: Update, context: CallbackContext) -> None:
    if len(context.args) == 0:
        update.message.reply_text("请提供搜索的内容，例如：/search 你想要搜索的文本")
        return

    # 获取用户输入的搜索文本
    search_text = " ".join(context.args)

    # sc改
    try:
        # 调用 get_recommendation 函数获取图片文件路径
        recommendation_result = get_recommendation(search_text) # dictionary

        if recommendation_result['flag'] == -1:
            update.message.reply_text("服务器出现问题，请稍后重试，错误原因（服务器未能远程连接到GPT服务器）")
        elif recommendation_result['flag'] == -2:
            update.message.reply_text("暂时没有相关推荐。")
        elif recommendation_result['flag'] == -3:
            update.message.reply_text("服务器数据库有问题，正在维护。")
        else:
            # 发送图片给用户
            with open(recommendation_result['img_path'], 'rb') as image_file:
                update.message.reply_photo(photo=image_file)

    except Exception as e:
        update.message.reply_text(f"发生错误: {str(e)}")


# ===================================================================================================

# sc改：用户可以选择自己上传图片进行genrage，或者根据之前search或者上一次generate的结果继续进行修改工作
# 处理 /generate 命令的函数  # 用户自己上传图片版本的处理生成
def generate(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    user_states[user_id] = {"stage": "generate"}  # 设置状态，等待图片上传
    update.message.reply_text("您可以上传一张衣服图片，或者用自然语言对上一次search或generate的结果进行进一步修改。")

''' 这个是你原来注释掉的东西，我给他搞成能折叠的注释形式 sc改
# # 处理上传图片和描述的函数
# def handle_photos_and_description(update: Update, context: CallbackContext) -> None:
#     user_id = update.message.from_user.id
    
#     # 检查用户是否在生成图片的流程中,不是的话触发GPT
#     if user_id not in user_states or user_states[user_id]["stage"] != "waiting_for_photos":
#         user_message = update.message.text
#         # 将用户消息发送给ChatGPT并获取回复
#         gpt_reply = chat_with_gpt(user_message)
#         # 将ChatGPT的回复发送给用户
#         update.message.reply_text(gpt_reply)
#         # return  # 如果用户不在生成流程中，则忽略该消息

#     # 处理用户发送的图片
#     elif update.message.photo:
#         # 初始化图片列表
#         if "photos" not in user_states[user_id]:
#             user_states[user_id]["photos"] = []

#         # 获取用户发送的图片
#         photos = update.message.photo
#         file = context.bot.get_file(photos[-1].file_id)
#         image_path = f"image_{len(user_states[user_id]['photos'])}.jpg"
#         file.download(image_path)

#         # 保存图片路径
#         user_states[user_id]["photos"].append(image_path)

#         if len(user_states[user_id]["photos"]) == 2:
#             update.message.reply_text("请提供您上传的衣服部件的名称。")
#         else:
#             update.message.reply_text("请上传衣服部件的图片。")

#     # 处理用户发送的描述
#     elif update.message.text and len(user_states[user_id]["photos"]) == 2:
#         description = update.message.text
#         update.message.reply_text("图片和描述已收到，正在生成图像，请稍等...")

#         # 调用生成图片的函数
#         try:
#             image_paths = user_states[user_id]["photos"]
#             generated_image_path = generate_images(image_paths[0], image_paths[1], description)
#             output_path = "generated_image.png"
#             generated_image_path.save(output_path)

#             # 发送生成的图片给用户
#             with open(output_path, 'rb') as image_file:
#                 update.message.reply_photo(photo=InputFile(image_file))

#             # 清理用户状态和临时文件
#             for path in image_paths:
#                 if os.path.exists(path):
#                     os.remove(path)
#             if os.path.exists(output_path):
#                 os.remove(output_path)

#             # 重置用户状态
#             del user_states[user_id]

#         except Exception as e:
#             update.message.reply_text(f"发生错误: {str(e)}")
'''


# 处理上传图片的函数 sc改
def handle_uploaded_photos(update: Update, context: CallbackContext, user_id: int) -> str:
    
    if "photos" not in user_states[user_id]:
        user_states[user_id]["photos"] = []
    if "texts" not in user_states[user_id]:
        user_states[user_id]["texts"] = []

    # 获取用户发送的图片
    photos = update.message.photo
    file = context.bot.get_file(photos[-1].file_id)
    image_path = f"image_{len(user_states[user_id]['photos'])}.jpg"
    
    # 保存图片路径
    file.download(image_path)
    user_states[user_id]["photos"].append(image_path)

    # 获取用户对所上传图片的描述
    if len(user_states[user_id]["photos"]) == 1:
        update.message.reply_text("请提供需要添加的部件，并告知部件颜色。")
        img_description = update.message.text
        user_states[user_id]["texts"].append(img_description)
        # # 把用户上传的图片和描述信息保存在历史记录中
        # insert_in_memory_table(img_description, "user upload", image_path)
    
    # if len(user_states[user_id]["texts"]) == 1:
    #     update.message.reply_text("您需要修改这件衣服的什么配件？请在'belt', 'body_piece', 'button', 'collar', 'cuff', 'hood', 'placket', 'pocket', 'waist_band'中选一个。")
    #     accessory_choice = update.message.text
    #     user_states[user_id]["texts"].append(accessory_choice)
    
    # if len(user_states[user_id]["texts"]) == 2:
    #     return {"path": image_path, "description":img_description, "accessory": accessory_choice}
    


# 处理生成图像的函数
def handle_image_generation(update: Update, context: CallbackContext, user_id: int, description: str) -> None:
    update.message.reply_text("图片和描述已收到，正在生成图像，请稍等...")

    try:
        image_paths = user_states[user_id]["photos"]
        generated_image_path = generate_images(image_paths[0], image_paths[1], description)
        output_path = "generated_image.png"
        generated_image_path.save(output_path)

        # 发送生成的图片给用户
        with open(output_path, 'rb') as image_file:
            update.message.reply_photo(photo=InputFile(image_file))

        # 清理用户状态和临时文件
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(output_path):
            os.remove(output_path)

        # 重置用户状态
        del user_states[user_id]

    except Exception as e:
        update.message.reply_text(f"发生错误: {str(e)}")


# 调用上述函数的逻辑
def handle_photos_and_description(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    
    # sc改
    # 检查用户是选择上传图片进行修改还是选择历史记录进行修改，对应两个不同的流程
    # or user_states[user_id]["stage"] != "generate" or user_states[user_id]["stage"] != "tryon"
    if user_id not in user_states:
        user_message = update.message.text
        # 将用户消息发送给ChatGPT并获取回复
        gpt_reply = chat_with_gpt(user_message)
        update.message.reply_text(gpt_reply)

    if user_states[user_id]["stage"] == "generate":
        # 处理用户发送的图片：说明走的是发图片->简单描述->选择修改的部件名称->生成图片并返回 的流程
        if update.message.photo:
            user_upload_dict = handle_uploaded_photos(update, context, user_id)
            '''
            sc: 暂时改到此为止
            '''
        # 处理用户发送的描述
        elif update.message.text and len(user_states[user_id]["photos"]) == 2:
            description = update.message.text
            handle_image_generation(update, context, user_id, description)

    elif user_states[user_id]["stage"] == "tryon":
        # 获取用户发送的图片
        if update.message.photo:
            photo = update.message.photo[-1]  # 获取分辨率最高的图片
            file = context.bot.get_file(photo.file_id)
            cloth_image_path = f"cloth_image_{user_id}.jpg"
            file.download(cloth_image_path)

            # 根据用户选择的性别设置模特图片路径
            gender = user_states[user_id]["gender"]
            image_path = f"../CatVTON/resource/demo/example/person/{gender}/model_5.png"
            output_dir = "tryonOutput"
            cloth_type = "upper"  # 服装类型

            update.message.reply_text("正在处理图片，请稍等...")

            try:
                # 调用 tryon_process 生成试穿结果
                tryon_process(image_path, cloth_image_path, output_dir, cloth_type)

                # 发送生成的试穿结果图片给用户
                result_path = os.path.join(output_dir, "result.png")
                with open(result_path, 'rb') as result_image:
                    update.message.reply_photo(photo=InputFile(result_image))

                # # 删除临时保存的图片文件
                # os.remove(cloth_image_path)
                # os.remove(result_path)

                # 重置用户状态
                del user_states[user_id]

            except Exception as e:
                update.message.reply_text(f"发生错误: {str(e)}")


# 处理 /tryon 命令的函数
def tryon(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    
    # 检查用户输入的命令参数是否为 'men' 或 'women'
    if len(context.args) != 1 or context.args[0] not in ['men', 'women']:
        update.message.reply_text("请输入 /tryon men 或 /tryon women 来选择模特。")
        return

    # 保存用户选择的模特性别
    user_states[user_id] = {
        "stage": "tryon",
        "gender": context.args[0]  # 保存用户选择的性别
    }

    # 提示用户上传图片
    update.message.reply_text(f"请上传一张衣服的图片，模特选择为 {context.args[0]}。")

# # 处理上传的衣服图片
# def handle_photo(update: Update, context: CallbackContext) -> None:
#     user_id = update.message.from_user.id
    
#     # 检查用户是否在等待上传图片的状态
#     if user_id not in user_states or user_states[user_id]["stage"] != "waiting_for_photo":
#         return  # 如果用户不在等待图片流程中，则忽略消息

#     # 获取用户发送的图片
#     if update.message.photo:
#         photo = update.message.photo[-1]  # 获取分辨率最高的图片
#         file = context.bot.get_file(photo.file_id)
#         cloth_image_path = f"cloth_image_{user_id}.jpg"
#         file.download(cloth_image_path)

#         # 根据用户选择的性别设置模特图片路径
#         gender = user_states[user_id]["gender"]
#         image_path = f"../CatVTON/resource/demo/example/person/{gender}/model_5.png"
#         output_dir = "tryonOutput"
#         cloth_type = "upper"  # 服装类型

#         update.message.reply_text("正在处理图片，请稍等...")

#         try:
#             # 调用 tryon_process 生成试穿结果
#             tryon_process(image_path, cloth_image_path, output_dir, cloth_type)

#             # 发送生成的试穿结果图片给用户
#             result_path = os.path.join(output_dir, "result.png")
#             with open(result_path, 'rb') as result_image:
#                 update.message.reply_photo(photo=InputFile(result_image))

#             # 删除临时保存的图片文件
#             os.remove(cloth_image_path)
#             os.remove(result_path)

#             # 重置用户状态
#             del user_states[user_id]

#         except Exception as e:
#             update.message.reply_text(f"发生错误: {str(e)}")



# # 处理用户消息的函数
# def handle_message(update: Update, context: CallbackContext) -> None:
#     user_message = update.message.text
#     update.message.reply_text("请先发送 /generate 来启动生成图片流程。")

def main():
    # 使用你的Telegram Bot API Token创建Updater
    updater = Updater("7881685894:AAFYvj03v8YAl5mruQwEGG2kWZH5D04FkaA", use_context=True)

    # 获取调度器来注册处理器
    dp = updater.dispatcher

    # 注册 /start 命令的处理器
    dp.add_handler(CommandHandler("start", start))

    # 注册 /search 命令的处理器
    dp.add_handler(CommandHandler("search", search))

    # 注册 /generate 命令的处理器
    dp.add_handler(CommandHandler("generate", generate))

    # 注册 /tryon 命令的处理器
    dp.add_handler(CommandHandler("tryon", tryon))

    # 注册处理图片和描述的处理器
    dp.add_handler(MessageHandler(Filters.photo | Filters.text, handle_photos_and_description))

    #  # 注册处理文本消息的处理器
    # dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message_gpt))


    # 启动机器人
    updater.start_polling()

    # 保持程序运行，直到手动停止
    updater.idle()

if __name__ == '__main__':
    main()
