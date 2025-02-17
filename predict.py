from sklearn.preprocessing import LabelEncoder
import time
from main import model, model_name, seed
import pandas as pd

data_test = pd.read_csv('public_test.csv')

type_housing = pd.unique(data_test['Тип_жилья'])
for i in range(len(type_housing)):
    data_test['Тип_жилья'] = data_test['Тип_жилья'].replace(type_housing[i], i + 1)

encoder = LabelEncoder()
data_test['Тип_жилья'] = encoder.fit_transform(data_test['Тип_жилья'])
data_test['Направление'] = encoder.fit_transform(data_test['Направление'])
data_test['Город'] = encoder.fit_transform(data_test['Город'])
data_test['Ктгр_энергоэффективности'] = encoder.fit_transform(data_test['Ктгр_энергоэффективности'])
data_test['Ктгр_вредных_выбросов'] = encoder.fit_transform(data_test['Ктгр_вредных_выбросов'])

X_test = data_test[
    ['Тип_жилья', 'Индекс', 'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото',
     'Нлч_гаража', 'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
     'Ктгр_вредных_выбросов',
     'Размер_участка', 'Нлч_балкона', 'Ктгр_энергоэффективности', 'Направление', 'Кво_спален',
     'Кво_ванных', 'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота']]

data_test.fillna(-1, inplace=True)

y_pred = model.predict(X_test)

data_test["Цена"] = y_pred

data_test[["id", "Цена"]].to_csv(f'public_test_predict_seed_{seed}_{model_name}_{str(time.time_ns())}.csv', index=False)
