import pickle
import train
import pandas as pd
path_audio_files='./Audios/'

if __name__ == "__main__":
     
    engine = train.bd_engine(driver='mysql', user='root', passwrd='sound123',
                       host='localhost', port=3306,db='soundKey')
    
    #connection = engine.connect()
    
    #stmt_user  = 'SELECT * FROM audiouser'
   
    #df_user    = pd.read_sql(connection.execute(stmt_user).fetchall(),engine)

    #recebendo o usuario
    df_user = pd.read_csv('./Database/audiouser.csv',header=None)
    df_user.columns = ['basid','basnome','userid']
    train.create_features_columns(df_user)
    #covertendo os arquivo de .3pg para wav
    train.convert_df_to_wav(df_user)
    #extraindo as features dos audios
    train.fill_features_columns(df_user)
    #creando a lista de features e target
    target,features=train.create_tg_features()
    #defindo aquivo de modelo
    model='./model/mlp.model'
    loaded_model = pickle.load(open(model, 'rb'))
    #criando as features
    
    pred = loaded_model.predict(df_user[features])
    return pred[0]==df_user[target]    
