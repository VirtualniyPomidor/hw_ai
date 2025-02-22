from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from main import model, model_name, seed
import pandas as pd
import os
import numpy as np

data_test = pd.read_csv('public_test.csv')

# Преобразуем категориальный признак в числовой с помощью LabelEncoder
encoder = LabelEncoder()
data_test['Тип_жилья'] = encoder.fit_transform(data_test['Тип_жилья'])
data_test['Направление'] = encoder.fit_transform(data_test['Направление'])
data_test['Город'] = encoder.fit_transform(data_test['Город'])
data_test['Ктгр_энергоэффективности'] = encoder.fit_transform(data_test['Ктгр_энергоэффективности'])
data_test['Ктгр_вредных_выбросов'] = encoder.fit_transform(data_test['Ктгр_вредных_выбросов'])

data_test['Соотношение_этажей'] = data_test['Этаж'] / data_test['Верхний_этаж']

# Логарифмирование числовых признаков с большим разбросом
data_test['Площадь_лог'] = np.log1p(data_test['Площадь'])

data_test.fillna(-1, inplace=True)

# Разделяем данные на признаки (X) и целевую переменную (y)
X_test = data_test[
    ['Площадь_лог', 'Тип_жилья', 'Индекс',
     'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото',
     'Нлч_гаража', 'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
     'Ктгр_вредных_выбросов',
     'Размер_участка', 'Нлч_балкона', 'Ктгр_энергоэффективности', 'Направление', 'Кво_спален',
     'Кво_ванных', 'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота']]

data_test.fillna(-1, inplace=True)

y_pred = model.predict(X_test)

data_test["Цена"] = y_pred

name = f'public_test_predict_seed_basic_{seed}_{model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'

path = 'tests'
os.makedirs(path, exist_ok=True)

full_path = os.path.join(path, name)

data_test[["id", "Цена"]].to_csv(full_path, index=False)

print(f"Файл сохранен: {full_path}")
