from flask import Flask, request, jsonify
import subprocess
import pandas as pd
from sqlalchemy import create_engine
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

# inputs
training_data = ['./Database/usuarios.csv','./Database/baseaudio.csv']

model_directory = './model'
model_file_name = '%s/mlp.model' % model_directory

@app.route('/predict')
def predict():
    import train
    engine = train.bd_engine(driver='mysql', user='root', passwrd='sound123',
                       host='localhost', port=3306,db='soundKey')
    
    #connection = engine.connect()
    
    #stmt_user  = 'SELECT * FROM audiouser'
   
    #df_user    = pd.read_sql(connection.execute(stmt_user).fetchall(),engine)

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
    pred = loaded_model.predict(df_user[features])
    return jsonify({'pred':pred.tolist()})


@app.route('/train', methods=['GET'])
def train():
    import train

    engine = bd_engine(driver='mysql', user='root', passwrd='sound123',
                       host='localhost', port=3306,db='soundKey')
    
    #connection = engine.connect()
    
    #stmt_usuarios   = 'SELECT * FROM usuarios'
    #stmt_baseaudios = 'SELECT * FROM baseaudio'
    
    #df_baseaudio  = pd.read_sql(connection.execute(stmt_baseaudios).fetchall(),engine) 
    #df_usuarios    = pd.read_sql(connection.execute(stmt_baseaudios).fetchall(),engine)

    
    df_usuarios  = pd.read_csv(training_data[0],header=None)
    df_usuarios.columns = ['userid','name','username','senha','usetokenexpire','usertoken']
    #base de audios para treino
    df_baseaudio = pd.read_csv(training_data[1],header=None)
    df_baseaudio.columns = ['basid','basnome','userid']
    df_merged      = pd.merge(df_usuarios,df_baseaudio,on='userid')
    #Creado a colunas de features
    train.create_features_columns(df_merged)
    #covertendo os arquivo de .3pg para wav
    train.convert_df_to_wav(df_merged)
    #extraindo as features dos audios
    train.fill_features_columns(df_merged)
    #creando a lista de features e target
    target,features=train.create_tg_features()

    scaler_x = MinMaxScaler(feature_range=(-0.8,0.8))
    x = df_merged[features]
    y = df_merged[target]
    scaler_x.fit(x)
    x = scaler_x.transform(x)

    

    # capture a list of columns that will be used for prediction
    mlp = MLPClassifier(hidden_layer_sizes=(2*len(features)+1), activation='logistic', solver='adam', alpha=0.0001, 
                        batch_size='auto',learning_rate='constant', learning_rate_init=0.09, power_t=0.5, max_iter=1000, shuffle=True, 
                        random_state=None, tol=0.00001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                        early_stopping=False, validation_fraction=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #treinando o classificado
    mlp = mlp.fit(x,y)
    #fazendo a predição
    #y_pred = mlp.predict(X_test)
    #salvando o modelo
    pickle.dump(mlp,open('./model/mlp.model','wb'))
    print ('Model training score: %s' % mlp.score(x, y))

    return 'Success'


# @app.route('/wipe', methods=['GET'])
# def wipe():
#     try:
#         shutil.rmtree('model')
#         os.makedirs(model_directory)
#         return 'Model wiped'

#     except Exception, e:
#         print str(e)
#         return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
        app.run(host='0.0.0.0', port=80, debug=True)
