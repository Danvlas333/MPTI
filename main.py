import os
import json
import re
import docx
from sentence_transformers import SentenceTransformer

def extract_events_precise(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    doc = docx.Document(file_path)
    lines = [para.text for para in doc.paragraphs if para.text.strip() != ""]

    MONTHS = [
        'январ[ья]', 'феврал[ья]', 'марта?', 'апрел[ья]', 'ма[йя]', 'июн[ья]',
        'июл[ья]', 'августа?', 'сентябр[ья]', 'октябр[ья]', 'ноябр[ья]', 'декабр[ья]'
    ]
    MONTH_PATTERN = '|'.join(MONTHS)

    date_pattern = re.compile(
        r'^\s*('
        r'\d{1,2}[./]\s*\d{1,2}(?:[./]\s*\d{2,4})?|'
        r'\d{1,2}\s*[–\-]\s*\d{1,2}\s+(?:' + MONTH_PATTERN + r')|'
        r'\d{1,2}\s+(?:' + MONTH_PATTERN + r')(?:\s+\d{4})?|'
        r'(?:' + MONTH_PATTERN + r')\s+\d{4}|'
        r'\d{1,2}[./]\d{1,2}\s*[–\-]\s*\d{1,2}[./]\d{1,2}|'
        r'\d{1,2}\s*[–\-]\s*\d{1,2}\s*(?:' + MONTH_PATTERN + r')'
        r')',
        re.IGNORECASE | re.UNICODE
    )

    year_pattern = re.compile(r'^\s*(2024|2025)\s*$', re.UNICODE)
    section_header_pattern = re.compile(r'^\s*Образовательная повестка,\s*\d{4}\s*$', re.IGNORECASE | re.UNICODE)

    events = []
    current_date = None
    current_lines = []
    current_year = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if section_header_pattern.match(stripped):
            continue

        year_match = year_pattern.match(stripped)
        if year_match:
            current_year = year_match.group(1)
            continue

        date_match = date_pattern.match(stripped)
        if date_match:
            if current_date is not None and current_lines:
                full_desc = ' '.join(current_lines).strip()
                events.append((current_date, full_desc))
                current_lines = []

            date_str = date_match.group(1).strip()
            desc_part = stripped[len(date_match.group(0)):].strip()
            desc_part = re.sub(r'^[-–—:\s]+', '', desc_part)

            if current_year and not re.search(r'\b20\d{2}\b', date_str):
                if re.search(r'[а-яё]', date_str, re.IGNORECASE):
                    date_str += f" {current_year}"
                elif '.' in date_str:
                    date_str += f".{current_year[2:]}"
                elif '/' in date_str:
                    date_str += f"/{current_year[2:]}"
                else:
                    date_str += f" {current_year}"

            current_date = date_str
            if desc_part:
                current_lines.append(desc_part)
        else:
            if current_date is not None:
                current_lines.append(stripped)

    if current_date is not None and current_lines:
        events.append((current_date, ' '.join(current_lines).strip()))

    return events

def clean_event_for_embedding(text: str) -> str:
    text = re.sub(r'\b\d+\s*(участник|человек|персон|чел\.?)\b', '', text, flags=re.IGNORECASE)
    parts = re.split(r'[–—:]', text, maxsplit=1)
    core = parts[0].strip()
    if len(core.split()) < 2:
        core = text.strip()
    return re.sub(r'\s+', ' ', core)

def main():
    DOCX_FILE = "Dop_materialy_AI_pomoshhnik_po_mediam_0a34958fc5.docx"
    JSON_OUTPUT = "events_vector_db.json"
    MODEL_NAME = "cointegrated/rubert-tiny2"

    print(" Извлечение событий...")
    events = extract_events_precise(DOCX_FILE)
    print(f" Найдено {len(events)} событий.")

    print(" Загрузка модели rubert-tiny2...")
    model = SentenceTransformer(MODEL_NAME)

    print(" Векторизация...")
    database = []
    for i, (date_str, raw_text) in enumerate(events):
        clean_text = clean_event_for_embedding(raw_text)
        vector = model.encode(clean_text, convert_to_numpy=True, normalize_embeddings=True)
        database.append({
            "date": date_str,
            "text": raw_text,
            "vector": vector.tolist()
        })
        print(f"  [{i+1}/{len(events)}] {clean_text[:50]}...")

    print(" Сохранение...")
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False, indent=2)

    print(f" База сохранена: {JSON_OUTPUT}")
    print(f" Размер вектора: {vector.shape[0]}")

if __name__ == "__main__":

    main()
