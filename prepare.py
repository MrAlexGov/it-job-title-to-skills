"""Сборка датасета «название IT-должности → ключевые навыки» из двух открытых выгрузок hh.ru.

Источники (скачиваются через Kaggle API, к hh.ru запросов нет):
  A. etietopabraham/jobs-raw-data           — все сферы, 04–05.2023, CC BY 4.0 → IT отбираем по названию
  B. ilyazawilsiv/it-vacancies-from-headhunter-website — IT, 09–10.2023, Apache-2.0 → по проф. роли hh
Запуск: python prepare.py  (ожидает raw/Raw_Jobs.csv и raw/IT_vacancies.csv)
"""
import collections
import json
import os
import re

import pandas as pd

MIN_SKILL_FREQ = 30   # навык попадает в словарь, если встречается хотя бы в стольких вакансиях

# --- отбор IT по названию (только для источника A, где проф. роли нет) ---
IT_TITLE = re.compile(
    r"программист|разработчик|developer|engineer|devops|\bsre\b|тестировщик|\bqa\b|аналитик|analyst|"
    r"data scien|machine learning|\bml\b|\bai\b|\bnlp\b|computer vision|software|backend|frontend|"
    r"front-end|back-end|fullstack|full stack|full-stack|веб|web|верстальщик|1с|1c|php|python|java|"
    r"golang|\bgo\b|c\+\+|c#|\.net|javascript|typescript|react|vue|angular|node|ios|android|kotlin|"
    r"swift|flutter|unity|системный администратор|сетевой инженер|\bdba\b|информационной безопасности|"
    r"pentest|кибербезопас|team ?lead|тимлид|техлид|tech lead|\bcto\b|scrum|product owner|продакт|"
    r"product manager|проджект|project manager|\bux\b|\bui\b|технический писатель|data engineer|"
    r"инженер данных|\betl\b|битрикс|bitrix|wordpress|gamedev|геймдизайн", re.I)
NOT_IT_TITLE = re.compile(
    r"продаж|по работе с клиентами|кредит|финансов|бухгалт|маркетолог|врач|водител|механик|электри|"
    r"строит|сварщ|кладовщ|оператор|экономической безопасности|инженер[- ](по охране|сметчик|конструктор|"
    r"технолог|энергетик|проектировщик|по ремонту|пто)", re.I)

# --- проф. роли hh, которые берём из источника B ---
IT_ROLES = {
    "Программист, разработчик", "Специалист технической поддержки", "Аналитик", "Руководитель проектов",
    "Системный администратор", "Тестировщик", "Специалист по информационной безопасности",
    "Менеджер продукта", "Системный аналитик", "Бизнес-аналитик", "Системный инженер",
    "Руководитель группы разработки", "Сетевой инженер", "DevOps-инженер", "BI-аналитик, аналитик данных",
    "Технический директор (CTO)", "Технический писатель", "Продуктовый аналитик", "Дата-сайентист",
    "Руководитель отдела аналитики", "Директор по информационным технологиям (CIO)", "Гейм-дизайнер",
}

# --- общие («soft») навыки и прочий шум: из меток убираем, модель должна советовать профессиональное ---
SOFT = {
    "работа в команде", "аналитическое мышление", "работа с большим объемом информации", "грамотная речь",
    "пользователь пк", "деловая коммуникация", "деловая переписка", "организаторские навыки",
    "деловое общение", "удаленная работа", "работа в условиях многозадачности", "ведение переговоров",
    "грамотность", "креативность", "подготовка презентаций", "сбор и анализ информации",
    "системное мышление", "обучение и развитие", "клиентоориентированность", "проведение презентаций",
    "ориентация на результат", "многозадачность", "телефонные переговоры", "коммуникабельность",
    "ответственность", "навыки презентации", "умение работать в условиях многозадачности",
    "грамотная устная и письменная речь", "опытный пользователь пк", "продвинутый пользователь пк",
    "руководство коллективом", "ориентация на клиента", "высшее образование", "стратегическое мышление",
    "планирование", "умение работать в коллективе", "поиск информации в интернет",
    "точность и внимательность к деталям", "управление временем", "умение работать в команде",
    "стрессоустойчивость", "навыки переговоров", "аналитический склад ума", "управленческие навыки",
    "internet", "умение принимать решения", "расстановка приоритетов", "уверенный пользователь пк",
    "лидерство", "it", "информационные технологии", "работа с документами", "обучаемость",
    "самостоятельность", "пунктуальность", "исполнительность", "внимательность", "инициативность",
    "честность", "порядочность", "дисциплинированность", "умение работать с людьми", "работа с людьми",
    "высокая работоспособность", "нацеленность на результат", "быстрая обучаемость",
    "мотивация персонала", "управление персоналом", "обучение персонала", "аналитика", "аналитические исследования",
}

# --- синонимы: приводим к одному написанию ---
ALIASES = {
    "html5": "html", "css3": "css", "js": "javascript", "jira": "atlassian jira",
    "confluence": "atlassian confluence", "excel": "ms excel", "photoshop": "adobe photoshop",
    "go": "golang", "vue": "vue.js", "с#": "c#", "1c: предприятие": "1с: предприятие 8",
    "1с: предприятие": "1с: предприятие 8", "1c: бухгалтерия": "1с: бухгалтерия", "1c erp": "erp-системы на базе 1с",
    "ms sql server": "ms sql", "rest api": "rest", "restful api": "rest", "spring": "spring framework",
    "django": "django framework", "веб-дизайн": "web-дизайн", "project management": "управление проектами",
    "проектный менеджмент": "управление проектами", "agile project management": "agile",
    "cистемы управления базами данных": "субд", "postgres": "postgresql", "k8s": "kubernetes",
    "machine learning": "машинное обучение", "ml": "машинное обучение",
}


def norm(s):
    s = re.sub(r"\s+", " ", s.strip()).lower()
    return ALIASES.get(s, s)


def clean(skills):
    out = {norm(x) for x in skills if 1 < len(x.strip()) < 60}
    return sorted(s for s in out if s not in SOFT)


def load_a():
    df = pd.read_csv("raw/Raw_Jobs.csv", sep=";", usecols=["title", "key_skills", "experience"],
                     on_bad_lines="skip", engine="python").dropna(subset=["title", "key_skills"])
    df = df[df.title.str.contains(IT_TITLE) & ~df.title.str.contains(NOT_IT_TITLE)]
    return pd.DataFrame({"title": df.title.str.strip(), "skills": df.key_skills.map(lambda s: clean(s.split(","))),
                         "experience": df.experience, "source": "jobs-raw-data-2023-04"})


def load_b():
    df = pd.read_csv("raw/IT_vacancies.csv", usecols=["name", "key_skills", "professional_roles_name", "experience"])
    df = df[df.professional_roles_name.isin(IT_ROLES)].dropna(subset=["name", "key_skills"])
    parse = lambda s: [a or b for a, b in re.findall(r'"([^"]+)"|([^,{}"]+)', s)]
    return pd.DataFrame({"title": df.name.str.strip(), "skills": df.key_skills.map(lambda s: clean(parse(s))),
                         "experience": df.experience, "source": "it-vacancies-2023-09"})


def main():
    df = pd.concat([load_a(), load_b()], ignore_index=True)
    df = df[df.skills.map(len) >= 2]
    df["key"] = df.title.str.lower() + "|" + df.skills.map("|".join)
    df = df.drop_duplicates("key")
    freq = collections.Counter(s for l in df.skills for s in l)
    vocab = [s for s, n in freq.most_common() if n >= MIN_SKILL_FREQ]
    keep = set(vocab)
    df["skills"] = df.skills.map(lambda l: [s for s in l if s in keep])
    df = df[df.skills.map(len) >= 1]
    stats = {"examples": len(df), "skills_in_vocab": len(vocab), "by_source": df.source.value_counts().to_dict(),
             "avg_skills_per_vacancy": round(df.skills.map(len).mean(), 2), "min_skill_freq": MIN_SKILL_FREQ,
             "soft_skills_removed": len(SOFT), "aliases": len(ALIASES)}
    os.makedirs("kdata", exist_ok=True)
    with open("kdata/it_title_skills.jsonl", "w") as f:
        for r in df.itertuples():
            f.write(json.dumps({"title": r.title, "skills": r.skills, "experience": r.experience, "source": r.source},
                               ensure_ascii=False) + "\n")
    json.dump(vocab, open("kdata/skill_vocab.json", "w"), ensure_ascii=False)
    json.dump(stats, open("kdata/stats.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
