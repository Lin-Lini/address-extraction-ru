# Проект: Извлечение и валидация адресов (Полуфинал Волга IT 2024)

Несколько вариантов реализации пайплайна на Python: regex, Natasha, RuBERT + CatBoost. 
Проект сделан в рамках полуфинала Volga IT 2024.

| Файл | Назначение |
|------|-----------|
| `src/dataset_regex_variant.py` | Создание обучающего датасета с помощью регулярных выражений (Вариант 1). |
| `src/dataset_natasha_variant.py` | Создание датасета с помощью библиотеки **Natasha** (Вариант 2). |
| `src/train_catboost_variant.py` | Обучение модели **CatBoost** на разметке (Вариант 1). |
| `src/train_rubert_variant.py` | Обучение классификатора на эмбеддингах **RuBERT** (Вариант 2). |

## Быстрый старт

```bash
pip install -r requirements.txt

# Запуск варианта с regex
python src/dataset_regex_variant.py  

# Запуск варианта с Natasha
python src/dataset_natasha_variant.py  

# Обучение CatBoost
python src/train_catboost_variant.py  

# Обучение RuBERT + CatBoost
python src/train_rubert_variant.py  
```
CSV‑файлы с данными кладём в папку **data/** (папка игнорируется Git).

## Требования
Список библиотек – в `requirements.txt`.

## Лицензия
MIT
