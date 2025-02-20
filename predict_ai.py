import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from train import seed  # Импорт seed из train.py
from datetime import datetime

# Загрузка данных
test_data = pd.read_csv('public_test.csv')

# Повтор предобработки
test_data['Соотношение_этажей'] = test_data['Этаж'] / (test_data['Верхний_этаж'] + 1e-6)
test_data['Площадь_лог'] = np.log1p(test_data['Площадь'])
test_data.fillna(-1, inplace=True)

# Применение тех же преобразований
features = [
    'Площадь_лог', 'Кво_комнат', 'Кво_ванных', 'Широта', 'Долгота',
    'Ктгр_энергоэффективности', 'Размер_участка', 'Соотношение_этажей',
    'Кво_спален', 'Кво_фото', 'Этаж', 'Город', 'Тип_жилья'
]

# Загрузка нормализатора и модели
scaler = joblib.load(f'scaler_{seed}.pkl')
model = tf.keras.models.load_model(f'nn_model_{seed}')

# Предсказание
X_test = scaler.transform(test_data[features])
preds_log = model.predict(X_test).flatten()
preds = np.expm1(preds_log)  # Обратное преобразование

# Сохранение
name = f'public_test_predict_seed_ai_{seed}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'

path = 'tests'
os.makedirs(path, exist_ok=True)

pd.DataFrame({'id': test_data.id, 'Цена': preds}).to_csv(name, index=False)