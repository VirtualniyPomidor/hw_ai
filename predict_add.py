import pandas as pd
import numpy as np
import joblib
from main_add import seed

# Загрузка тестовых данных
test_data = pd.read_csv('public_test.csv')

# Предобработка как в train.py
cat_features = ['Тип_жилья', 'Город', 'Ктгр_энергоэффективности',
               'Ктгр_вредных_выбросов', 'Направление']

# Применение тех же преобразований
test_data['Соотношение_этажей'] = test_data['Этаж'] / (test_data['Верхний_этаж'] + 1e-6)
test_data['Эффективность_энергии'] = test_data['Ктгр_энергоэффективности'] / (test_data['Расход_тепла'] + 1)
test_data['Площадь_лог'] = np.log1p(test_data['Площадь'])
test_data.fillna(-1, inplace=True)

# Загрузка моделей
stack_model = joblib.load(f'stack_{seed}.pkl')

# Предсказание
X_test = test_data[features]
preds_log = stack_model.predict(X_test)
preds = np.expm1(preds_log)  # Обратное преобразование

# Сохранение
name = f'public_test_predict_seed_{seed}_{model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
pd.DataFrame({'id': test_data.id, 'Цена': preds}).to_csv(f'submission_{seed}.csv', index=False)