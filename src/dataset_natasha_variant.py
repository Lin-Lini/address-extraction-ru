import pandas as pd
import re
from natasha import Segmenter, MorphVocab, NewsNERTagger, NewsEmbedding, Doc
from transformers import BertTokenizer

# Загрузка данных
addresses = pd.read_csv('volgait2024-semifinal-addresses.csv', sep=';')
tasks = pd.read_csv('volgait2024-semifinal-task.csv', sep=';')

# Настройка Natasha и токенайзера RuBERT
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

# Функция для извлечения адресов с помощью NER
def extract_addresses_with_ner(comment):
    doc = Doc(comment)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    addresses = [span.text for span in doc.spans if span.type == 'LOC']
    return addresses

# Функция для сопоставления извлеченных адресов с полными адресами
def match_address(extracted_addresses):
    matched_uuids = []
    for address in extracted_addresses:
        best_match = addresses[addresses['house_full_address'].str.contains(address, case=False, regex=False)]
        if not best_match.empty:
            matched_uuids.append(best_match['house_uuid'].values[0])
    return matched_uuids

# Создание датасета с токенизацией для CatBoost и RuBERT
dataset = []
for _, row in tasks.iterrows():
    shutdown_id = row['shutdown_id']
    comment = row['comment']

    # Извлечение адресов из комментария
    extracted_addresses = extract_addresses_with_ner(comment)

    # Сопоставление извлеченных адресов с UUID домов
    uuids = match_address(extracted_addresses)

    # Токенизация для CatBoost (простое разбиение по словам)
    catboost_tokens = re.findall(r'\b\w+\b', comment.lower())

    # Токенизация для RuBERT
    rubert_tokens = tokenizer.encode(comment, add_special_tokens=True, max_length=512, truncation=True)

    # Формируем запись для датасета
    dataset.append({
        'shutdown_id': shutdown_id,
        'comment': comment,
        'extracted_addresses': ', '.join(extracted_addresses),
        'matched_uuids': ', '.join(uuids) if uuids else '',
        'catboost_tokens': ' '.join(catboost_tokens),  # Преобразуем список в строку для сохранения
        'rubert_tokens': rubert_tokens  # Список токенов для RuBERT
    })

# Преобразование в DataFrame и сохранение в CSV
df_dataset = pd.DataFrame(dataset)
df_dataset.to_csv('dataset_with_tokens.csv', index=False, sep=';')