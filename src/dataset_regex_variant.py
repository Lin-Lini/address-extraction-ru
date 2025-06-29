import pandas as pd
import re
from transformers import BertTokenizer
from tqdm import tqdm
from joblib import Parallel, delayed

# Загрузка данных
tasks = pd.read_csv('volgait2024-semifinal-task.csv', delimiter=';')
houses = pd.read_csv('volgait2024-semifinal-addresses.csv', delimiter=';')

# Инициализация токенайзера RuBERT
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

# Обновленное регулярное выражение для поиска улиц и номеров домов
full_pattern = r"(?:ул\.?|улица|пр\.?|проспект|ш\.?|п\.?|пер\.?|площадь|б-р\.?|мкр\.?|поселок|пос\.?)?\s*([А-ЯЁ][а-яё\s\-]+)\s*([\d\s,.-]+)"

# Преобразуем названия улиц в нижний регистр для оптимизации
houses['house_full_address'] = houses['house_full_address'].str.lower()


# Функция для извлечения адресов из комментария
def extract_addresses(comment):
    comment = clean_comment(comment)
    comment = re.sub(r'(\d+)-\s*[^\d\s]*', r'\1', comment)  # удаление ненужного текста после тире
    matches = re.findall(full_pattern, comment)

    extracted = []
    for match in matches:
        street_name = match[0]
        house_numbers = []
        for num in re.split(r'[\s,]+', match[1].strip()):
            if '-' in num:
                house_numbers.extend(expand_house_range(num))
            elif num.isdigit():
                house_numbers.append(num)

        if house_numbers:
            extracted.append((street_name.strip().lower(), house_numbers))  # Приведение улицы к нижнему регистру

    return extracted


# Функция для очистки комментариев
def clean_comment(comment):
    return re.sub(r"ХВС|Д=\d{3}", "", comment)


# Функция для развертывания диапазона номеров
def expand_house_range(house_range):
    try:
        start, end = map(int, house_range.split('-'))
        return list(range(start, end + 1))
    except ValueError:
        return []


# Функция сопоставления извлеченного адреса с полным адресом
def match_address(extracted_addresses, houses_df):
    matched_uuids = []
    for street, house_numbers in extracted_addresses:
        matches = houses_df[houses_df['house_full_address'].str.contains(street, case=False, regex=False)]
        for _, row in matches.iterrows():
            full_address = row['house_full_address']
            house_uuid = row['house_uuid']
            if any(f" {num} " in full_address or f" {num}," in full_address or f" {num}-" in full_address for num in
                   house_numbers):
                matched_uuids.append(house_uuid)
                break  # Останавливаемся, если нашли совпадение
    return matched_uuids


# Основная функция для обработки комментариев с параллельной обработкой
def process_comment(row):
    shutdown_id = row['shutdown_id']
    comment = row['comment']

    # Извлечение и сопоставление адресов
    extracted_addresses = extract_addresses(comment)
    matched_uuids = match_address(extracted_addresses, houses)

    # Токенизация для RuBERT
    rubert_tokens = tokenizer.encode(comment, add_special_tokens=True, max_length=512, truncation=True)

    # Токенизация для CatBoost
    catboost_label = 1 if matched_uuids else 0

    return {
        'shutdown_id': shutdown_id,
        'comment': comment,
        'extracted_addresses': ', '.join([f"{addr[0]} {', '.join(map(str, addr[1]))}" for addr in extracted_addresses]),
        'house_uuids': ', '.join(matched_uuids),
        'rubert_tokens': rubert_tokens,
        'catboost_label': catboost_label
    }


# Параллельная обработка комментариев и формирование датасета
dataset = Parallel(n_jobs=-1, backend="threading")(
    delayed(process_comment)(row) for _, row in tqdm(tasks.iterrows(), total=tasks.shape[0]))

# Преобразование в DataFrame и сохранение
df_dataset = pd.DataFrame(dataset)
df_dataset.to_csv('volgait2024-semifinal-result.csv', index=False)