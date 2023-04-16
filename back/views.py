from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from .utils import  numpy_data

## ИИ ге керек кітапханалаар
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam


# Загружаем данные о погоде
data = pd.read_csv("C:\\Users\\admin\\Desktop\\Nurik\\weather.csv")
# Create your views here.

class WeatherSunay():
    def prognoz_na_vlazhnosty_i_vremenih_otrezkov():
        #Прогноз на 12:00 15:00 18:00 21:00
        # Предобработка данных
        data = data.dropna()
        data = data[['Humidity', 'Weather', 'Date']]
        data = data.set_index('Date')
        data = data.pivot(columns='Weather')
        data = data['Humidity'][['Cloudy', 'Rainy', 'Sunny']]
        data = data.resample('D').mean().dropna()
        data = data.join(data.shift(-1), rsuffix='_tomorrow')
        data = data[['Cloudy', 'Rainy', 'Sunny', 'Cloudy_tomorrow', 'Rainy_tomorrow', 'Sunny_tomorrow']]
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data, columns=['Cloudy', 'Rainy', 'Sunny', 'Cloudy_tomorrow', 'Rainy_tomorrow', 'Sunny_tomorrow'])

        # Разделение данных на тренировочный и тестовый наборы
        X = data[['Cloudy', 'Rainy', 'Sunny']]
        y = data[['Cloudy_tomorrow', 'Rainy_tomorrow', 'Sunny_tomorrow']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Создание модели нейронной сети LSTM
        model = Sequential()
        model.add(LSTM(units=128, input_shape=(X_train.shape[1], 1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=3))

        # Компиляция модели
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Обучение модели
        history = model.fit(X_train.values.reshape(-1, 3, 1), y_train.values, epochs=100, batch_size=32, validation_split=0.2)

        # Проверка производительности модели
        y_pred = model.predict(X_test.values.reshape(-1, 3, 1))
        mse = mean_squared_error(y_test, y_pred)
        print('Mean Squared Error:', mse)

        # Визуализация результатов
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

        # Прогнозирование погоды на завтра
        today_weather = np.array([[data['Cloudy'][0], data['Rainy'][0], data['Sunny'][0]]])
        today_weather = scaler.transform(today_weather)
        today_weather = model.predict(today_weather)
        today_weather = scaler.inverse_transform(today_weather)
        return today_weather

class WeatherAnalysic(generics.GenericAPIView):
    def prognoz_na_zavtra():
        try:
              # Предобработка данных
            data = data.dropna()
            data = data[['MinTemp', 'MaxTemp']]
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            data = pd.DataFrame(data, columns=['MinTemp', 'MaxTemp'])

            # Разделение данных на тренировочный и тестовый наборы
            X = data[['MinTemp']]
            y = data['MaxTemp']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Создание модели нейронной сети LSTM
            model = Sequential()
            model.add(LSTM(units=128, input_shape=(X_train.shape[1], 1), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=64, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=32, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            # Компиляция модели
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

            # Обучение модели
            history = model.fit(X_train.values.reshape(-1, 1, 1), y_train.values, epochs=100, batch_size=32, validation_split=0.2)

            # Проверка производительности модели
            y_pred = model.predict(X_test.values.reshape(-1, 1, 1))
            mse = mean_squared_error(y_test, y_pred)
            print('Mean Squared Error:', mse)
            # Прогнозирование максимальной температуры влажности и на сегодня и н завтра 
            today_min_temp = np.array([[data['MinTemp'][0]]])
            today_min_temp = today_min_temp.reshape(-1, 1, 1)
            tomorrow_max_temp = model.predict(today_min_temp)
            tomorrow_max_temp = scaler.inverse_transform(tomorrow_max_temp)
            print('Прогноз максимальной температуры на завтра:', tomorrow_max_temp)
            return tomorrow_max_temp
        except:
            return "Болжау кезіндегі қателік"
        


class AiWeatherView(generics.GenericAPIView):
    permission_classes = (AllowAny,)

    def post(self, request):
        print(numpy_data(request.data['how_to_learn']))
        return Response(numpy_data(request.data['how_to_learn']),status=status.HTTP_200_OK)
