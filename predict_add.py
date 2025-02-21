import pandas as pd
import numpy as np
import joblib
from main_add import seed
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

# seed = 55749

# Загрузка тестовых данных
test_data = pd.read_csv('public_test.csv')

# Предобработка как в train.py
cat_features = ['Тип_жилья', 'Город', 'Ктгр_энергоэффективности',
                'Ктгр_вредных_выбросов', 'Направление']

features = [
    'Площадь_лог', 'Тип_жилья', 'Индекс',
    'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото', 'Нлч_гаража',
    'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
    'Ктгр_вредных_выбросов', 'Размер_участка', 'Нлч_балкона',
    'Ктгр_энергоэффективности', 'Направление', 'Кво_спален', 'Кво_ванных',
    'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота'
]

encoder = LabelEncoder()
for col in cat_features:
    test_data[col] = encoder.fit_transform(test_data[col])

# Применение тех же преобразований
test_data['Соотношение_этажей'] = test_data['Этаж'] / (test_data['Верхний_этаж'] + 1e-6)
test_data['Площадь_лог'] = np.log1p(test_data['Площадь'])
test_data.fillna(-1, inplace=True)

# Загрузка моделей
stack_model = joblib.load(f'stack_{seed}.pkl')
cat_model = joblib.load(f'cat_{seed}.pkl')
xgb_model = joblib.load(f'xgb_{seed}.pkl')
models = [stack_model, cat_model, xgb_model]
# Предсказание
for model in models:
    X_test = test_data[features]
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)  # Обратное преобразование

    # Сохранение
    name = f'public_test_predict_seed_add_{seed}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    path = 'tests'
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, name)

    pd.DataFrame({'id': test_data.id, 'Цена': preds}).to_csv(full_path, index=False)

    print(f"Файл сохранен: {full_path}")
