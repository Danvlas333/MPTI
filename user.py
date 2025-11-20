import json
import numpy as np
import re
import urllib3
from sentence_transformers import SentenceTransformer

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

QUERY = "Петербурге"
DB_PATH = "events_vector_db.json"
TOP_K = 30
MODEL_NAME = "cointegrated/rubert-tiny2"

USE_GIGACHAT = True
GIGACHAT_TOKEN = "MDE5YTlkYTItODZjYi03MjVjLTkwMjYtZjZmNWE3ZmIxNTBjOmViZmVkYTc0LWJhNjMtNGFmZS05MmY3LTdmOWVkODExZWE3Zg=="

NORTHWEST_CITIES = [
    "санкт-петербург", "спб", "петербург", "деловой петербург", "питер",
    "всеволожск", "гатчина", "каменногорск", "кириши", "кольцово", "луза",
    "выборг", "тосно", "волхов", "сосновый бор",
    "петрозаводск", "кондопога", "беломорск", "олонец",
    "мурманск", "апатиты", "ковдор", "мончегорск", "полярные зори",
    "архангельск", "новодвинск", "коряжма", "котлас", "нарьян-мар",
    "калининград", "черняховск", "гусев", "балтийск", "советск",
    "великий новгород", "новгород", "боровичи", "старая русса",
    "псков", "великие луки", "остров", "невель",
    "вологда", "череповец", "грязовец", "кириллов",
]

def load_vector_db(path):
    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)
    for item in db:
        item["vector"] = np.array(item["vector"], dtype=np.float32)
    return db

def is_date_query(query: str) -> bool:
    q = query.strip().lower()
    return bool(
        re.search(r'\b\d{1,2}\s+(январ|феврал|март|апрел|май|июн|июл|август|сентябр|октябр|ноябр|декабр)', q) or
        re.search(r'^\d{1,2}[./]\d{1,2}$', q)
    )

def extract_northwest_geo_hints(query: str):
    query_norm = query.lower().replace("-", " ")
    if "калининрад" in query_norm:
        query_norm = query_norm.replace("калининрад", "калининград")
    matches = []
    for city in NORTHWEST_CITIES:
        city_norm = city.lower().replace("-", " ")
        if city_norm in query_norm:
            matches.append(city)
    return matches

def find_similar_events(query: str, db_path: str, top_k: int = 5):
    model = SentenceTransformer(MODEL_NAME)
    db = load_vector_db(db_path)

    if is_date_query(query):
        results = []
        query_low = query.lower()
        for item in db:
            event_str = f"{item['date']} {item['text']}".lower()
            if (
                (re.search(r'\b(\d{1,2})\s+(январ|феврал|март|апрел|май|июн|июл|август|сентябр|октябр|ноябр|декабр)', query_low) and
                 any(month in event_str for month in ['январ', 'феврал', 'март', 'апрел', 'май', 'июн', 'июл', 'август', 'сентябр', 'октябр', 'ноябр', 'декабр']) and
                 re.search(r'\b' + re.escape(query_low.split()[0]) + r'\b', event_str))
                or
                (re.search(r'^\d{1,2}[./]\d{1,2}$', query_low) and query_low.replace('/', '.') in event_str)
            ):
                results.append({"date": item["date"], "text": item["text"], "score": 0.999})
        if results:
            return results[:top_k]

    query_vec = model.encode(query, normalize_embeddings=True)
    scores = [(float(np.dot(query_vec, item["vector"])), item) for item in db]
    scores.sort(key=lambda x: x[0], reverse=True)

    return [{
        "date": item["date"],
        "text": item["text"],
        "score": round(sim, 4)
    } for sim, item in scores[:top_k * 2]]

def filter_with_gigachat(query: str, candidates: list, credentials: str):
    if not candidates:
        return candidates

    try:
        from gigachat import GigaChat

        query_lower = query.lower()
        has_event_type = any(
            typ in query_lower
            for typ in [
                "встреча", "премия", "стратегическая сессия", "стратсессия", "хакатон",
                "лекция", "серия лекций", "митап", "визит", "круглый стол", "конкурс",
                "форум", "референс-визит", "межотраслевая сессия",
                "мероприятие с участием делегации", "конференция", "семинар"
            ]
        )

        events_block = "\n".join([
            f"ИНДЕКС: {i}\nДАТА: {item['date']}\nТЕКСТ: {item['text']}\n---"
            for i, item in enumerate(candidates)
        ])

        prompt = (
            "Ты — строгий фильтр релевантности. ТВОЯ ЗАДАЧА — вернуть ТОЛЬКО индексы событий, соответствующих запросу.\n\n"
            
            "1. АНАЛИЗ ЗАПРОСА:\n"
            "- Если в запросе есть **тип мероприятия** (например: «хакатоны», «митапы», «форум»), то событие должно **содержать это слово в любом падеже**.\n"
            "  Пример: запрос «хакатоны» → допустимо: «хакатон», «на хакатоне». Недопустимо: «сессия», «клуб», «AI-мероприятие», «конкурс».\n"
            "- Если в запросе есть **город** («Петербург», «СПб», «Калининград» и т.д.), событие должно происходить в этом городе или содержать упоминание связанной организации («СПбГУ», «Калининградский университет»).\n"
            "- Дата учитывается только при точном совпадении.\n\n"
            
            "2. ПРАВИЛА:\n"
            "- НЕ интерпретируй. НЕ делай аналогий. НЕ расширяй понятия.\n"
            "- Если тип мероприятия указан, но ни одно событие его не содержит — верни ПУСТОЙ список.\n"
            "- Если тип НЕ указан — НЕ фильтруй по типу.\n\n"
            
            "3. ФОРМАТ ОТВЕТА:\n"
            "Верни ТОЛЬКО валидный JSON в формате:\n"
            '{"relevant_events": [0, 2, 5]}\n'
            "Если ни одно событие не подходит — верни:\n"
            '{"relevant_events": []}\n'
            "НИКАКИХ пояснений, markdown, текста, объектов. Только JSON.\n\n"
            
            "ЗАПРОС:\n«{query}»\n\n"
            "СОБЫТИЯ:\n{events}\n\n"
            "ОТВЕТ:"
        ).format(query=query, events=events_block)

        with GigaChat(credentials=credentials, verify_ssl_certs=False) as giga:
            response = giga.chat(prompt)

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) > 2:
                raw = "\n".join(lines[1:-1])
            else:
                raw = ""

        raw = raw.strip()
        if not raw:
            return [] if has_event_type else candidates
        
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [] if has_event_type else candidates

        if not isinstance(parsed, dict):
            return [] if has_event_type else candidates

        indices = parsed.get("relevant_events")
        if indices is None or not isinstance(indices, list):
            return [] if has_event_type else candidates

        valid_indices = [i for i in indices if isinstance(i, int)]

        result = []
        for i in valid_indices:
            if 0 <= i < len(candidates):
                result.append(candidates[i])
        return result

    except Exception:
        query_lower = query.lower()
        has_type = any(
            typ in query_lower
            for typ in ["хакатон", "митап", "форум", "премия", "лекция", "сессия", "конференция", "семинар"]
        )
        return [] if has_type else candidates

if __name__ == "__main__":
    matches = find_similar_events(QUERY, DB_PATH, top_k=TOP_K)

    geo_hints = extract_northwest_geo_hints(QUERY)
    if geo_hints:
        filtered_geo = []
        for item in matches:
            event_text = (item["date"] + " " + item["text"]).lower()
            if any(city.lower() in event_text for city in geo_hints):
                filtered_geo.append(item)
        matches = filtered_geo

    if USE_GIGACHAT:
        matches = filter_with_gigachat(QUERY, matches, GIGACHAT_TOKEN)
        print(matches)
        for i, m in enumerate(matches[:TOP_K], 1):
            print(f"{i}. [{m['score']:.3f}] {m['date']} — {m['text']}")
    pass