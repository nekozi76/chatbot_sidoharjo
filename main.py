from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters
from bot import chatbot_response

async def handle_message(update: Update, context):
    user_input = update.message.text
    response = chatbot_response(user_input)  # Panggil fungsi chatbot yang sudah ada
    # Pastikan response adalah string, bukan array, atau ambil elemen pertama jika array
    if isinstance(response, list):
        response = response[0]
    await update.message.reply_text(response)

app = ApplicationBuilder().token("6721366616:AAFOL2MFskLFTHJzZCZMomF156lavlXoOXQ").build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
app.run_polling()
