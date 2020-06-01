from flask import Flask, render_template, request, session, send_from_directory, request, redirect
import pickle
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os, glob
import librosa
import mysql.connector as mysql
import subprocess

def extractFeature(file):
    data , sr = librosa.load(file)
    hasil=np.array([])
    mfccs=np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    hasil=np.hstack((hasil, mfccs))
    stft=np.abs(librosa.stft(data))
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    hasil=np.hstack((hasil, chroma))
    return hasil


app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = './storage'

@app.route('/')
def home():
    return render_template('upload.html')


# storage as static dir
@app.route('/storage/<path:x>')
def storage(x):
    return send_from_directory('storage', x)

# upload
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method =='POST':
        f = request.files['fileku']
        # print(f)
        # print(f.filename)
        fname = secure_filename(f.filename)
        print(fname)
        f.save(os.path.join(
            app.config['UPLOAD_FOLDER'],
            fname)
            )
        # files
        file = fname.split('.')
        dst = file[0]+".wav"
        sourceFile = "D:\\belajar IT\\Purwadhika JC Data Science\\Code\\Final Project\\storage\\"+fname
        endFile = "D:\\belajar IT\\Purwadhika JC Data Science\\Code\\Final Project\\storage\\"+dst
        if sourceFile != endFile:
            appDir=r"c:\\ffmpeg\\bin"
            os.chdir(appDir)
            os.system(f'ffmpeg -i "{sourceFile}" -acodec pcm_u8 -ar 22050 "{endFile}"')
            os.remove(sourceFile)
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
        # cursor.execute('create database databaseVoice')
        cursor.execute('use databaseVoice')
        # cursor.execute('DROP TABLE IF EXISTS voice')
        # cursor.execute(f"""create table voice (no int auto_increment primary key,
        # v0 float(50),v1 float(50),v2 float(50),v3 float(50),v4 float(50),v5 float(50),
        # v6 float(50),v7 float(50),v8 float(50),v9 float(50),v10 float(50),
        # v11 float(50),v12 float(50),v13 float(50),v14 float(50),v15 float(50),
        # v16 float(50),v17 float(50),v18 float(50),v19 float(50),v20 float(50),
        # v21 float(50),v22 float(50),v23 float(50),v24 float(50),v25 float(50),
        # v26 float(50),v27 float(50),v28 float(50),v29 float(50),v30 float(50),
        # v31 float(50),v32 float(50),v33 float(50),v34 float(50),v35 float(50),
        # v36 float(50),v37 float(50),v38 float(50),v39 float(50),v40 float(50),
        # v41 float(50),v42 float(50),v43 float(50),v44 float(50),v45 float(50),
        # v46 float(50),v47 float(50),v48 float(50),v49 float(50),v50 float(50),
        # v51 float(50), emotions varchar(100))""")

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
        return render_template('hasil.html', x=x)

if __name__ == '__main__':
    with open('FinalRandom', 'rb') as myModel:
        model = pickle.load(myModel)
    app.run(debug=True)
