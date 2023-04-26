import flask
from flask import render_template
import pickle
import sklearn  #1.1.2
from sklearn.ensemble import RandomForestRegressor
import tensorflow  #2.12.0
from tensorflow import keras


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':

        #Загружаем модель для предсказания первого целевого признака "Модуль упругости при растяжении, ГПа"
        with open('../models/pipe_rfr_elasticity.pkl', 'rb') as f:
            loaded_model_elasticity = pickle.load(f)

        #Загружаем модель для предсказания первого целевого признака "Модуль упругости при растяжении, ГПа" 
        loaded_model_strength = keras.models.load_model('../models/NN_strength')


        #Заполняем датасет данными, поступающими с веб-формы
        data_elasticity = []  #Хранилище для данных
        for i in range(11):  #По каждому из 11 признаков
            data_elasticity.append(float(flask.request.form[f"feature{i+1}"]))  #Читаем данные, поступающие с веб-формы


        #Делаем предсказание первого целевого признака с использованием загруженной модели случайного леса
        y_pred_elasticity = loaded_model_elasticity.predict([data_elasticity])

        #Делаем предсказание второго целевого признака с использованием искусственной нейронной сети
        y_pred_strength = loaded_model_strength.predict([data_elasticity])


        #Отображаем форму с результатами для признаков "Модуль упругости при растяжении, ГПа" и "Прочность при растяжении, МПа"
        return render_template('main.html', 
        elasticity_result=round(y_pred_elasticity[0], 5), 
        strength_result=round(y_pred_strength[0][0], 5))


if __name__ == '__main__':
    app.run()
    