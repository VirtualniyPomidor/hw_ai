import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from main_ai import seed
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import os
import warnings

# Загрузка данных
test_data = pd.read_csv('public_test.csv')

# Применение тех же преобразований
features = [
    'Площадь_лог', 'Тип_жилья', 'Индекс',
    'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото', 'Нлч_гаража',
    'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
    'Ктгр_вредных_выбросов', 'Размер_участка', 'Нлч_балкона',
    'Ктгр_энергоэффективности', 'Направление', 'Кво_спален', 'Кво_ванных',
    'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота'
]

cat_features = ['Тип_жилья', 'Город', 'Ктгр_энергоэффективности',
                'Ктгр_вредных_выбросов', 'Направление']

encoder = LabelEncoder()
for col in cat_features:
    test_data[col] = encoder.fit_transform(test_data[col])

# Повтор предобработки
test_data['Соотношение_этажей'] = test_data['Этаж'] / (test_data['Верхний_этаж'] + 1e-6)
test_data['Площадь_лог'] = np.log1p(test_data['Площадь'])
test_data.fillna(-1, inplace=True)

# Загрузка нормализатора и модели
scaler = joblib.load(f'scaler_{seed}.pkl')
model = tf.keras.models.load_model(f'nn_model_{seed}.keras')

# Предсказание
X_test = scaler.transform(test_data[features])
preds_log = model.predict(X_test).flatten()
preds = np.expm1(preds_log)  # Обратное преобразование

# Сохранение
name = f'public_test_predict_seed_ai_{seed}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
path = 'tests'
os.makedirs(path, exist_ok=True)
full_path = os.path.join(path, name)

pd.DataFrame({'id': test_data.id, 'Цена': preds}).to_csv(full_path, index=False)

print(f"Файл сохранен: {full_path}")