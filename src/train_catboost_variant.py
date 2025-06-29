import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
import torch

# Загрузка данных
df = pd.read_csv('volgait2024-semifinal-result.csv')

# Инициализация токенайзера и модели RuBERT
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
model.eval()  # Установка модели в режим оценки

# Функция для преобразования токенов в усредненные векторы
def get_average_vector(tokens):
    with torch.no_grad():
        inputs = torch.tensor(tokens).unsqueeze(0)  # добавление размерности батча
        outputs = model(inputs)[0]  # Получение выходных данных модели
        return outputs.mean(dim=1).numpy()[0]  # Усреднение векторов токенов

# Преобразование токенов в усредненные векторы
df['average_vectors'] = df['rubert_tokens'].apply(eval).apply(get_average_vector)

# Разделение данных на обучающую и валидационную выборки
X = np.array(list(df['average_vectors']))
y = df['catboost_label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Создание Pool объектов для CatBoost
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

# Инициализация модели CatBoost
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    eval_metric='F1',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

# Обучение модели
model.fit(train_pool, eval_set=val_pool)

# Предсказание на валидационной выборке
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Оценка метрик
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred_proba)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

# Визуализация важности признаков
feature_importances = model.get_feature_importance()
sns.barplot(x=feature_importances, y=list(range(len(feature_importances))))
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature Index")
plt.show()

# Сохранение модели
model.save_model('catboost_model.cbm')