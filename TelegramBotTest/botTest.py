from openai import OpenAI
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# 设置OpenAI的API密钥


# 定义 /start 命令的处理函数
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("你好！我是一个智能聊天机器人，向我发送任何问题，我将会帮你回答！")

# 定义与OpenAI ChatGPT进行对话的函数
def chat_with_gpt(message: str) -> str:
    try:
        # 使用聊天接口生成回复
        client = OpenAI(
           api_key = '我们的那个API'
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

# 处理用户消息的函数
def handle_message(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    # 将用户消息发送给ChatGPT并获取回复
    gpt_reply = chat_with_gpt(user_message)
    # 将ChatGPT的回复发送给用户
    update.message.reply_text(gpt_reply)

def main():
    # 使用你的Telegram Bot API Token创建Updater
    updater = Updater("7881685894:AAFYvj03v8YAl5mruQwEGG2kWZH5D04FkaA", use_context=True)

    # 获取调度器来注册处理器
    dp = updater.dispatcher

    # 注册 /start 命令的处理器
    dp.add_handler(CommandHandler("start", start))

    # 注册处理文本消息的处理器
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    # 启动机器人
    updater.start_polling()

    # 保持程序运行，直到手动停止
    updater.idle()

if __name__ == '__main__':
    main()
