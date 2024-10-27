from telegram import Update, InputFile
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from openai import OpenAI
import os
import sys
sys.path.append("E:/CodeAllNight/CETA/CETA/") 
# from database.insert_images import insert_in_memory_table
from databaseAPI.get_recommendation import get_recommendation
from fashion_adapter.image_generation import generate_images  # 导入生成图像的函数
# sys.path.append("E:/CodeAllNight/CETA/CETA/tryon")
from try_on import tryon_process
from databaseAPI.add_sector import add_sector_by_uploading
# 状态跟踪字典
user_states = {}

# 定义 /start 命令的处理函数
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("""
Hello! I'm your AI fashion design assistant. I can provide fashion design advice, offer classic clothing styles, provide masks for clothing accessories (recommended for advanced users), generate new clothing designs through AI, and create AI-generated model try-on images of your clothes. The specific features are as follows:

- Send a text message directly: Chat with our fashion design assistant to help you design clothes that you are satisfied with.
- /start: View the bot help.
- /search: Let our AI fashion design assistant recommend some clothing for you.
- /searchmask: Search for clothing accessory masks for AI generation (recommended for advanced users).
- /generate: Try the amazing clothing generation feature.
- /tryon: Try on your favorite or AI-generated clothes on our model.
""")



# 定义与OpenAI ChatGPT进行对话的函数
def chat_with_gpt(message: str) -> str:
    try:
        # 使用聊天接口生成回复
        print("Sending message to OpenAI API...")
        client = OpenAI(
           api_key = 'You own ChatGPT API key'
        )
        response  = client.chat.completions.create(
            messages=[
                 {"role": "system", "content": "You are a helpful cloth design assistant."},  # 系统消息设定助手行为
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
        update.message.reply_text("Please provide the search content, for example: /search I want a white shirt.")
        return

    # 获取用户输入的搜索文本
    search_text = " ".join(context.args)

    try:
        # 调用 get_recommendation 函数获取图片文件路径
        recommendation_result = get_recommendation(search_text)
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
        # if recommendation_result == -1:
        #     update.message.reply_text("没有找到匹配的结果，请提供更多详细信息。")
        # else:
        #     # 发送图片给用户
        #     with open(recommendation_result, 'rb') as image_file:
        #         update.message.reply_photo(photo=image_file)

    except Exception as e:
        update.message.reply_text(f"An error occurred:{str(e)}")

# 添加 /searchmask 命令的处理函数
def searchmask(update: Update, context: CallbackContext) -> None:
    if len(context.args) == 0:
        update.message.reply_text("Please provide a search description, for example: /searchmask I want a black jacket with a hood.")
        return

    # 获取用户输入的搜索描述
    search_text = " ".join(context.args)

    try:
        # 调用 add_sector_by_uploading 函数
        result_dict = add_sector_by_uploading(search_text)

        if result_dict['flag'] == -1:
            update.message.reply_text("There is an issue with the server. Please try again later. Error reason: (The server failed to connect remotely to the Chat-GPT server).")
        elif result_dict['flag'] == -2:
            update.message.reply_text("There are no recommendations at the moment.")
        elif result_dict['flag'] == -31:
            update.message.reply_text(f"File not found: {result_dict['selected_image_path']}")
        elif result_dict['flag'] == -32:
            update.message.reply_text(f"File not found: {result_dict['selected_mask_path']}")
        else:
            # 发送掩码图片给用户
            with open(result_dict['selected_mask_path'], 'rb') as image_file:
                update.message.reply_photo(photo=image_file)

    except Exception as e:
        update.message.reply_text(f"An error occurred: {str(e)}")


# 处理 /generate 命令的函数  # 用户自己上传图片版本的处理生成
def generate(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    user_states[user_id] = {"stage": "generate"}  # 设置状态，等待图片上传
    update.message.reply_text("Please upload an image of the clothing first.")


# 处理上传图片的函数
def handle_uploaded_photos(update: Update, context: CallbackContext, user_id: int) -> None:
    if "photos" not in user_states[user_id]:
        user_states[user_id]["photos"] = []

    # 获取用户发送的图片
    photos = update.message.photo
    file = context.bot.get_file(photos[-1].file_id)
    image_path = f"image_{len(user_states[user_id]['photos'])}.jpg"
    file.download(image_path)

    # 保存图片路径
    user_states[user_id]["photos"].append(image_path)

    if len(user_states[user_id]["photos"]) == 2:
        update.message.reply_text("Please provide the name of the clothing item you uploaded.")
    else:
        update.message.reply_text("Please upload an image of the clothing part or describe in text the part you want to add to the clothing.")


# 处理生成图像的函数
def handle_image_generation(update: Update, context: CallbackContext, user_id: int, description: str) -> None:
    update.message.reply_text("Image and description received, generating the image, please wait...")
    print(description)

    if len(user_states[user_id]["photos"]) == 2:
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
            update.message.reply_text(f"An error occurred: {str(e)}")
    
    elif len(user_states[user_id]["photos"]) == 1:
        try:
            image_paths = user_states[user_id]["photos"]
            result_dict = add_sector_by_uploading(description)
            # print(result_dict['selected_mask_path'])
            mask_file = result_dict['selected_mask_path']
            mask = result_dict['accessory']
            generated_image_path = generate_images(image_paths[0], mask_file, mask)
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
            update.message.reply_text(f"An error occurred: {str(e)}")

# 调用上述函数的逻辑
def handle_photos_and_description(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    
    # 检查用户是否在生成图片的流程中,不是的话触发GPT
    # or user_states[user_id]["stage"] != "generate" or user_states[user_id]["stage"] != "tryon"
    if user_id not in user_states:
        user_message = update.message.text
        # 将用户消息发送给ChatGPT并获取回复
        gpt_reply = chat_with_gpt(user_message)
        update.message.reply_text(gpt_reply)

    if user_states[user_id]["stage"] == "generate":
        # 处理用户发送的图片
        if update.message.photo:
            handle_uploaded_photos(update, context, user_id)

        elif update.message.text and len(user_states[user_id]["photos"]) == 1:
            description = update.message.text
            handle_image_generation(update, context, user_id, description)

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

            update.message.reply_text("Processing the image, please wait...")

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
                update.message.reply_text(f"An error occurred: {str(e)}")


# 处理 /tryon 命令的函数
def tryon(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    
    # 检查用户输入的命令参数是否为 'men' 或 'women'
    if len(context.args) != 1 or context.args[0] not in ['men', 'women']:
        update.message.reply_text("Please enter /tryon men or /tryon women to select a model.")
        return

    # 保存用户选择的模特性别
    user_states[user_id] = {
        "stage": "tryon",
        "gender": context.args[0]  # 保存用户选择的性别
    }

    # 提示用户上传图片
    update.message.reply_text(f"Please upload an image of the clothing, the selected model is {context.args[0]}。")


def main():
    # 使用你的Telegram Bot API Token创建Updater
    updater = Updater("7881685894:AAFYvj03v8YAl5mruQwEGG2kWZH5D04FkaA", use_context=True)

    # 获取调度器来注册处理器
    dp = updater.dispatcher

    # 注册 /start 命令的处理器
    dp.add_handler(CommandHandler("start", start))

    # 注册 /search 命令的处理器
    dp.add_handler(CommandHandler("search", search))

    # 注册 /searchmask 命令的处理器
    dp.add_handler(CommandHandler("searchmask", searchmask))

    # 注册 /generate 命令的处理器
    dp.add_handler(CommandHandler("generate", generate))

    # 注册 /tryon 命令的处理器
    dp.add_handler(CommandHandler("tryon", tryon))

    # 注册处理图片和描述的处理器
    dp.add_handler(MessageHandler(Filters.photo | Filters.text, handle_photos_and_description))

    # 启动机器人
    updater.start_polling()

    # 保持程序运行，直到手动停止
    updater.idle()

if __name__ == '__main__':
    main()
