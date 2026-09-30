# IT job title → skills (rubert-tiny2)

Маленькая модель (29M параметров) по **названию IT-должности** предсказывает, какие **ключевые навыки**
работодатели указывают в таких вакансиях. Например, «Backend-разработчик Go» → golang 0.98, postgresql 0.54,
docker 0.41, git 0.39, linux 0.27, kubernetes 0.25, redis 0.20.

Демо в браузере (ONNX, без сервера): [Space](https://huggingface.co/spaces/MrAlexGov/it-job-title-to-skills) · Модель: [HF](https://huggingface.co/MrAlexGov/it-job-title-to-skills-rubert-tiny2) ·
Датасет: [MrAlexGov/it-job-title-skills-ru](https://huggingface.co/datasets/MrAlexGov/it-job-title-skills-ru)

```python
# pip install torch transformers huggingface_hub; файл skills_model.py лежит в этом репозитории
from skills_model import SkillRecommender
rec = SkillRecommender("MrAlexGov/it-job-title-to-skills-rubert-tiny2")
rec.recommend("Data Scientist", top_k=8)
# [('python', 0.93), ('машинное обучение', 0.62), ('sql', 0.61), ('data science', 0.43),
#  ('pandas', 0.41), ('numpy', 0.31), ('data analysis', 0.16), ('математическая статистика', 0.15)]
```

## Данные
Две открытые выгрузки hh.ru, скачанные через Kaggle API (к hh.ru не было ни одного запроса):

| Источник | Период | Лицензия | Как отобраны IT-вакансии | Примеров |
|---|---|---|---|---|
| [etietopabraham/jobs-raw-data](https://www.kaggle.com/datasets/etietopabraham/jobs-raw-data) | 04–05.2023, все сферы | CC BY 4.0 | регулярные выражения по названию (+ стоп-слова: «продаж», «механик»…) | 20 360 |
| [ilyazawilsiv/it-vacancies-from-headhunter-website](https://www.kaggle.com/datasets/ilyazawilsiv/it-vacancies-from-headhunter-website) | 09–10.2023, IT | Apache-2.0 | профессиональные роли hh (22 IT-роли) | 30 669 |

Обработка (`prepare.py`):
- навыки приведены к нижнему регистру, 28 синонимов склеены (html5 → html, js → javascript, k8s → kubernetes…);
- **71 общий навык удалён** из меток («работа в команде», «грамотная речь», «пользователь ПК»…): модель должна
  советовать профессиональные навыки, а не общие фразы;
- дубли (одинаковые название + набор навыков) удалены; в словарь вошли навыки, встретившиеся ≥ 30 раз.

Итого **51 029 вакансий, 1 119 навыков**, в среднем 5.4 навыка на вакансию.

## Модель и обучение (`train_skills.py`)
Энкодер `cointegrated/rubert-tiny2` + mean pooling + линейная голова на 1 119 выходов (sigmoid, BCE).
Полное дообучение: AdamW (энкодер 5e-5, голова 1e-3), batch 64, max_len 32, fp16, до 8 эпох с выбором
лучшей по MAP@50 на валидации. 2×T4 на Kaggle, **2.2 минуты**.

## Оценка
**Разбиение по уникальным названиям** (GroupShuffleSplit): в тесте только те названия, которых не было в
обучении. Иначе модель просто запомнила бы «Системный администратор», и метрики были бы завышены.
Train 37 457 / test 10 156.

| Метод | P@5 | R@5 | P@10 | R@10 | MAP@50 |
|---|---|---|---|---|---|
| Самые популярные навыки | 0.107 | 0.102 | 0.086 | 0.168 | 0.105 |
| kNN по TF-IDF символьных n-грамм названия (k=50) | 0.306 | 0.317 | 0.220 | 0.441 | 0.344 |
| ruBert-base (178M), дообучение | 0.336 | 0.351 | 0.242 | 0.489 | 0.383 |
| **rubert-tiny2 (29M), дообучение — эта модель** | **0.345** | **0.363** | **0.247** | **0.500** | **0.394** |

- Модель лучше сильного бейзлайна kNN на 14% по MAP@50; крупная ruBert-base не дала выигрыша (названия
  короткие, в 4.5 раза дольше обучение).
- P@5 = 0.35: в среднем 1.7 из 5 предложенных навыков есть **в конкретной** вакансии. Разные работодатели на
  одну должность пишут разные наборы, поэтому это нижняя оценка полезности рекомендаций.
- Первая версия (только первый источник, общие навыки в метках): tiny2 MAP@50 0.322 против kNN 0.290.

## Файлы
`model.safetensors` + `head.pt` + `skill_vocab.json` — PyTorch (через `skills_model.py`);
`model_full_int8.onnx` — энкодер + пулинг + голова одним графом, int8 (для браузера/onnxruntime);
`prepare.py`, `train_skills.py`, `metrics.json`, `stats.json` — вся логика и результаты.

## Сравнение с большими LLM (Kaggle Benchmarks model proxy)
Те же 200 невиданных названий (277 вакансий). LLM просили выдать 20 навыков в стиле поля «Ключевые навыки» hh.ru;
ответы сопоставлены со словарём (нормализация + синонимы + нечёткое совпадение ≥ 92). **strict** — несопоставленные
навыки считаются промахами, **lenient** — выкидываются (без штрафа за формулировки). Код: `bench/`.

| Модель | MAP@20 strict | MAP@20 lenient | P@5 lenient | навыков LLM в словаре |
|---|---|---|---|---|
| **rubert-tiny2 (эта модель, 29M)** | **0.383** | **0.383** | **0.350** | 100% |
| TF-IDF kNN | 0.340 | 0.340 | 0.313 | 100% |
| Claude Sonnet 5 | 0.318 | 0.335 | 0.318 | 84% |
| Gemini 3 Flash | 0.323 | 0.333 | 0.325 | 87% |
| Gemini 3.1 Flash Lite | 0.311 | 0.321 | 0.308 | 84% |
| DeepSeek R1 | 0.279 | 0.290 | 0.291 | 82% |
| Qwen3-Next-80B | 0.277 | 0.288 | 0.290 | 83% |
| GPT-5.4 nano | 0.251 | 0.264 | 0.266 | 80% |

LLM советуют разумные навыки «в целом», но хуже предсказывают, что работодатели реально пишут в вакансиях hh.
Эндпоинт `ibm/granite-4.0-h-small` исключён: по тексту ошибки прокси он обслуживался Qwen3-0.6B; gpt-oss-120b —
3/200 ответов из-за перегрузки прокси.

## Ограничения
- **Данные 2023 года.** Новых ролей модель не знает: «Специалист по промпт-инжинирингу» → ms excel 0.22,
  ms powerpoint 0.18, заключение договоров 0.15 — низкие вероятности и общие офисные навыки. Для AI-ролей 2025+
  нужны свежие данные. Низкая максимальная вероятность (< 0.3) — признак, что должность модели незнакома.
- Отбор IT по названию (источник A) неидеален, часть шума осталась.
- Часть синонимов не склеена (react / react.js / react/redux).
- Вероятность — оценка того, что навык укажут в вакансии, а не «насколько навык важен».
- Отражает требования работодателей на hh.ru (Россия), а не рынок в целом.

Первоисточник данных — hh.ru; указаны авторы выгрузок и лицензии.

## Как воспроизвести
```bash
pip install -r requirements.txt kaggle
kaggle datasets download etietopabraham/jobs-raw-data --unzip -p raw
kaggle datasets download ilyazawilsiv/it-vacancies-from-headhunter-website --unzip -p raw
python prepare.py              # → kdata/it_title_skills.jsonl, skill_vocab.json, stats.json
# train_skills.py рассчитан на Kaggle GPU (ищет данные в /kaggle/input); ~2 мин на T4
python skills_model.py "Backend-разработчик Go"   # инференс готовой модели с HF
```
`demo/index.html` — статическое демо: onnxruntime-web + токенизатор transformers.js, модель грузится с HF.

## Структура
| Файл | Что делает |
|---|---|
| `prepare.py` | отбор IT-вакансий, чистка и нормализация навыков, словарь |
| `train_skills.py` | сплит по названиям, бейзлайны (популярность, TF-IDF kNN), дообучение rubert-tiny2 и ruBert-base, метрики |
| `skills_model.py` | класс `SkillRecommender` для инференса |
| `metrics.json` | все метрики последнего прогона + примеры |
| `demo/index.html` | демо в браузере |
