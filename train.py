#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 18:09:36 2018

@author: leandro
"""

import subprocess
import numpy as np
from scipy import signal
import librosa
import pandas as pd
from sqlalchemy import create_engine
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

path_audio_files='/var/www/html/uploads/'

def convert_to_wav(file,out_format='.wav'):
    command = 'ffmpeg -i '+ file + ' -vn '+ file[:-4] + out_format
    subprocess.call(command, shell=True)

def convert_df_to_wav(dataframe,inp_fomart='.3gp',out_format='.wav',column='basnome'):
    for file in dataframe[column].tolist():
        if file[-4:]==inp_fomart:
            convert_to_wav(path_audio_files+file,out_format)
            #atulizando nome dos arquivos
    dataframe[column] = dataframe[column].str.replace(inp_fomart,out_format)
    
def log_specgram(audio, sample_rate, window_size=20,step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,fs=sample_rate, window='hann',
                                            nperseg=nperseg,noverlap=noverlap,detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def create_tg_features():
    target= ['userid']
    features = ['melspectrogra_mean','melspectrogra_max', 'melspectrogra_min','melspectrogra_std',
                'mfcc_mean', 'mfcc_max', 'mfcc_min','mfcc_std',
                'rmse_mean', 'rmse_max', 'rmse_std',
                'chroma_stft_mean', 'chroma_stft_max', 'chroma_stft_std',
                'bandwidth_mean', 'bandwidth_max', 'bandwidth_std',
                'centroid_mean', 'centroid_max', 'centroid_std']
    return target,features

def bd_engine(driver='myssql',user='root',passwrd=None,host='locahost',port=80,db=None):
    return create_engine(driver + '://' + user + ':' + passwrd + '@' + host + ':' + str(port) + '/' +db)

def create_features_columns(dataframe):
    
    dataframe['melspectrogra_mean']=np.nan
    dataframe['melspectrogra_max' ]=np.nan
    dataframe['melspectrogra_min' ]=np.nan
    dataframe['melspectrogra_std' ]=np.nan

    dataframe['mfcc_mean']=np.nan
    dataframe['mfcc_max' ]=np.nan
    dataframe['mfcc_min' ]=np.nan
    dataframe['mfcc_std' ]=np.nan

    dataframe['rmse_mean']=np.nan
    dataframe['rmse_max' ]=np.nan
    dataframe['rmse_min' ]=np.nan
    dataframe['rmse_std' ]=np.nan

    dataframe['chroma_stft_mean']=np.nan
    dataframe['chroma_stft_max' ]=np.nan
    dataframe['chroma_stft_min' ]=np.nan
    dataframe['chroma_stft_std' ]=np.nan
    
    dataframe['bandwidth_mean']=np.nan
    dataframe['bandwidth_max' ]=np.nan
    dataframe['bandwidth_min' ]=np.nan
    dataframe['bandwidth_std' ]=np.nan
    
    dataframe['centroid_mean']=np.nan
    dataframe['centroid_max' ]=np.nan
    dataframe['centroid_min' ]=np.nan
    dataframe['centroid_std' ]=np.nan
    
def fill_features_columns(dataframe):
    
    for i,file in enumerate(dataframe['basnome'].tolist()):
        data, sample_rate = librosa.load(path_audio_files + file)
        freqs, times, spectrogram = log_specgram(data, sample_rate)
        
        S = librosa.feature.melspectrogram(data, sr=sample_rate, n_mels=128)
        log_S = librosa.power_to_db(data, ref=np.max)
        mfcc = librosa.feature.mfcc(data,sr=sample_rate,S=S, n_mfcc=128)
        #zero_crossing_rate = librosa.feature.zero_crossing_rate(data)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(data,sr=sample_rate,S=S)
        spectral_centroid = librosa.feature.spectral_centroid(data,sr=sample_rate,S=S)
        rmse = librosa.feature.rmse(data)
        chroma_stft = librosa.feature.chroma_stft(data,sr=sample_rate,S=S)
        
        dataframe.loc[i,'melspectrogra_mean']=log_S.mean()
        dataframe.loc[i,'melspectrogra_max']=log_S.max()
        dataframe.loc[i,'melspectrogra_min']=log_S.min()
        dataframe.loc[i,'melspectrogra_std']=log_S.std()
        
        dataframe.loc[i,'mfcc_mean']=mfcc.mean()
        dataframe.loc[i,'mfcc_max']=mfcc.max()
        dataframe.loc[i,'mfcc_min']=mfcc.min()
        dataframe.loc[i,'mfcc_std']=mfcc.std()
        
        dataframe.loc[i,'rmse_mean']=rmse.mean()
        dataframe.loc[i,'rmse_max']=rmse.max()
        dataframe.loc[i,'rmse_min']=rmse.min()
        dataframe.loc[i,'rmse_std']=rmse.std()
        
        dataframe.loc[i,'chroma_stft_mean']=chroma_stft.mean()
        dataframe.loc[i,'chroma_stft_max']=chroma_stft.max()
        dataframe.loc[i,'chroma_stft_min']=chroma_stft.min()
        dataframe.loc[i,'chroma_stft_std']=chroma_stft.std()
        
        dataframe.loc[i,'bandwidth_mean']=spectral_bandwidth.mean()
        dataframe.loc[i,'bandwidth_max']=spectral_bandwidth.max()
        dataframe.loc[i,'bandwidth_min']=spectral_bandwidth.min()
        dataframe.loc[i,'bandwidth_std']=spectral_bandwidth.std()
        
        dataframe.loc[i,'centroid_mean']=spectral_centroid.mean()
        dataframe.loc[i,'centroid_max']=spectral_centroid.max()
        dataframe.loc[i,'centroid_min']=spectral_centroid.min()
        dataframe.loc[i,'centroid_std']=spectral_centroid.std()

if __name__ == "__main__":
    
    engine = bd_engine(driver='mysql', user='root', passwrd='sound123',
                       host='localhost', port=3306,db='soundKey')
    
    #connection = engine.connect()
    
    #stmt_usuarios   = 'SELECT * FROM usuarios'
    #stmt_baseaudios = 'SELECT * FROM baseaudio'
    
    #df_baseaudio  = pd.read_sql(connection.execute(stmt_baseaudios).fetchall(),engine) 
    #df_usuarios    = pd.read_sql(connection.execute(stmt_baseaudios).fetchall(),engine)

    #base de usuarios#base de 
    df_usuarios  = pd.read_csv('./Database/usuarios.csv',header=None)
    df_usuarios.columns = ['userid','name','username','senha','usetokenexpire','usertoken']
    #base de audios para treino
    df_baseaudio = pd.read_csv('./Database/baseaudio.csv',header=None)
    df_baseaudio.columns = ['basid','basnome','userid']
    df_merged      = pd.merge(df_usuarios,df_baseaudio,on='userid')
    #Creado a colunas de features
    create_features_columns(df_merged)
    #covertendo os arquivo de .3pg para wav
    convert_df_to_wav(df_merged)
    #extraindo as features dos audios
    fill_features_columns(df_merged)
    #creando a lista de features e target
    target,features=create_tg_features()
    #Fazendo o Split
    X_train, X_test, y_train, y_test = train_test_split(df_merged[features], 
                                                    df_merged[target],test_size=.5)
    #normalizando a base
    scaler_x = MinMaxScaler(feature_range=(-0.8,0.8))

    scaler_x.fit(X_train)
    X_train = scaler_x.transform(X_train)
    X_test  = scaler_x.transform(X_test)
    #Criando o classificador
    mlp = MLPClassifier(hidden_layer_sizes=(2*len(features)+1), activation='logistic', solver='adam', alpha=0.0001, 
                      batch_size='auto',learning_rate='constant', learning_rate_init=0.09, power_t=0.5, max_iter=100, shuffle=True, 
                       random_state=None, tol=0.00001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                       early_stopping=False, validation_fraction=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #treinando o classificado
    mlp = mlp.fit(X_train,y_train.values.ravel())
    #fazendo a predição
    y_pred = mlp.predict(X_test)
    #salvando o modelo
    pickle.dump(mlp,open('./model/mlp.model','wb'))





        
