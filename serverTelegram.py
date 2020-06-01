import logging
import mysql.connector.locales.eng.client_error
import mysql.connector as mysql
import telegram
from telegram import ReplyKeyboardMarkup
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters,
                          ConversationHandler)
import pandas as pd
import numpy as np
import os, glob
import librosa
import pickle

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

CHOOSING_FIRST, TYPING_CHOICE = range(2)

reply_keyboard = [['Voice Classifier']]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)

def facts_to_str(user_data):
    facts = list()
    for key, value in user_data.items():
        facts.append('{} - {}'.format(key, value))
    return "\n".join(facts).join(['\n', '\n'])

def extractFeature(file):
    data , sr = librosa.load(file)
    hasil=np.array([])
    mfccs=np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    hasil=np.hstack((hasil, mfccs))
    stft=np.abs(librosa.stft(data))
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    hasil=np.hstack((hasil, chroma))
    return hasil

def start(update, context):
    usernameData = update.message['chat']['username']
    userChatid = update.message['chat']['id']
    update.message.reply_text(
    "Hi! My name is Voice Classifier. I will classify your voice. ",reply_markup=markup)
    return CHOOSING_FIRST

def voice(update, context):
    usernameData = update.message['chat']['username']
    userChatid = update.message['chat']['id']
    text = update.message.text
    context.user_data['choice'] = text
    update.message.reply_text(
        'Silahkan rekam / kirim suara anda !')
    return TYPING_CHOICE

def received_voice(update, context):

    try:
        fname = 'voice_file'
        voice_file = update.message.voice.get_file()
        fileExt = update.message.voice.mime_type[-1:-4]
        voice_file.download(custom_path="D:\\belajar IT\\Purwadhika JC Data Science\\Code\\Final Project\\storage\\voice_file"+fileExt)
        dst = fname+".wav"
        sourceFile = "D:\\belajar IT\\Purwadhika JC Data Science\\Code\\Final Project\\storage\\voice_file"+fileExt
    except:
        fname = update.message.document.file_name
        file = fname.split('.')
        dst = file[0]+".wav"
        voice_file = update.message.document.get_file()
        voice_file.download(custom_path="D:\\belajar IT\\Purwadhika JC Data Science\\Code\\Final Project\\storage\\"+fname)
        sourceFile = "D:\\belajar IT\\Purwadhika JC Data Science\\Code\\Final Project\\storage\\"+fname


    endFile = "D:\\belajar IT\\Purwadhika JC Data Science\\Code\\Final Project\\storage\\"+dst

    if sourceFile != endFile:
        appDir=r"c:\\ffmpeg\\bin"
        os.chdir(appDir)
        os.system(f'ffmpeg -i "{sourceFile}" -acodec pcm_u8 -ar 22050 "{endFile}"')
        os.remove(sourceFile)

    with open('D:\\belajar IT\\Purwadhika JC Data Science\\Code\\Final Project\\FinalRandom', 'rb') as myModel:
        model = pickle.load(myModel)

    powerranger = extractFeature("D:\\belajar IT\\Purwadhika JC Data Science\\Code\\Final Project\\storage\\"+dst)
    df = pd.DataFrame(powerranger).T
    x = model.predict([powerranger])
    x = str(x[0]).upper()


    df['emotions'] = x
    dbku = mysql.connect(
         host = 'localhost',
         port = 3307,
         user = 'root'
    #      auth_plugin = 'mysql_native_password'
    )

    cursor = dbku.cursor()
    cursor.execute('use databaseVoice')


    class NumpyMySQLConverter(mysql.conversion.MySQLConverter):
    # A mysql.connector Converter that handles Numpy types
        def _float32_to_mysql(self, value):
            return float(value)
        def _float64_to_mysql(self, value):
            return float(value)
        def _int32_to_mysql(self, value):
            return int(value)
        def _int64_to_mysql(self, value):
            return int(value)

    dbku.set_converter_class(NumpyMySQLConverter)
    for i in range(len(df)):
        listofTuple = tuple([i for i in df.iloc[i]])
        queryku = '''insert into voice (v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41,v42,v43,v44,v45,v46,v47,v48,v49,v50,v51, emotions)
                    values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
        cursor.execute(queryku, listofTuple)
        dbku.commit()
    os.remove(endFile)

    update.message.reply_text(f"Voice Classifier for this voice is {x}",
                            reply_markup=markup)
    return CHOOSING_FIRST


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def done(update, context):
    user_data = context.user_data
    if 'choice' in user_data:
        del user_data['choice']

    update.message.reply_text("I learned these facts about you:"
                              "{}"
                              "Until next time!".format(facts_to_str(user_data)))

    user_data.clear()
    return ConversationHandler.END

def main():
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater("1139326088:AAHTSKrrxI2V-atx9bxSz4lRkgppDjEAD34", use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Add conversation handler with the states CHOOSING, TYPING_CHOICE and TYPING_REPLY
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSING_FIRST: [MessageHandler(Filters.regex('^(Voice Classifier)$'),
                                                 voice)
                       ],

            TYPING_CHOICE: [MessageHandler(Filters.all,
                                           received_voice)
                            ],
             },
        fallbacks=[MessageHandler(Filters.regex('^Done$'), done)]
    )

    dp.add_handler(conv_handler)
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
