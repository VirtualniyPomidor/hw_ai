import pandas as pd
import numpy as np
import joblib
from main_light import seed
from datetime import datetime
import os

test_data = pd.read_csv('public_test.csv')

features = ['Тип_жилья', 'Индекс',
            'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото', 'Нлч_гаража',
            'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
            'Ктгр_вредных_выбросов', 'Размер_участка', 'Нлч_балкона',
            'Ктгр_энергоэффективности', 'Направление', 'Кво_спален', 'Кво_ванных',
            'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота']

cat_features = ['Тип_жилья', 'Город', 'Ктгр_энергоэффективности',
                'Ктгр_вредных_выбросов', 'Направление']

for col in cat_features:
    test_data[col] = test_data[col].astype('category')

for col in features:
    if col in cat_features:
        test_data[col] = test_data[col].cat.add_categories('unknown')  # Добавляем 'unknown' в категории
        test_data[col] = test_data[col].fillna('unknown')
    else:
        test_data[col] = test_data[col].fillna(test_data[col].median())

model = joblib.load(f'lgb_model_{seed}.pkl')

X_test = test_data[features]
preds_log = model.predict(X_test)
preds = np.expm1(preds_log)

# Сохранение
name = f'public_test_predict_seed-{seed}_light_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
path = 'tests'
os.makedirs(path, exist_ok=True)
full_path = os.path.join(path, name)

pd.DataFrame({'id': test_data.id, 'Цена': preds}).to_csv(full_path, index=False)

print(f"Файл сохранен: {full_path}")