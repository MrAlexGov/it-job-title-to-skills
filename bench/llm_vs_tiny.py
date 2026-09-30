"""Бенчмарк: rubert-tiny2 против LLM (через Kaggle Benchmarks model proxy) на задаче «должность → навыки».
Тестовые названия — те же, что в train_skills.py (GroupShuffleSplit, seed 42), выборка N уникальных названий."""
import json, os, re, sys, time, concurrent.futures as cf
import numpy as np
from rapidfuzz import process, fuzz
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, "..")
from prepare import norm, SOFT
from skills_model import SkillRecommender
import kaggle_benchmarks as kbench

N_TITLES = int(os.environ.get("N_TITLES", 200)); K_LLM = 20; SEED = 42
rows = [json.loads(l) for l in open("../kdata/it_title_skills.jsonl")]
vocab = json.load(open("../kdata/skill_vocab.json")); idx = {s: i for i, s in enumerate(vocab)}
titles = [r["title"] for r in rows]; groups = [t.lower().strip() for t in titles]
Y = np.zeros((len(rows), len(vocab)), dtype=np.float32)
for i, r in enumerate(rows):
    for s in r["skills"]: Y[i, idx[s]] = 1
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(titles, groups=groups))
test_titles = sorted({groups[i] for i in te})
rng = np.random.RandomState(SEED); sample = sorted(rng.choice(test_titles, N_TITLES, replace=False))
by_title = {}
for i in te:
    if groups[i] in set(sample): by_title.setdefault(groups[i], []).append(i)
orig = {g: titles[by_title[g][0]] for g in sample}   # исходное написание названия
print(f"{len(sample)} названий, {sum(map(len, by_title.values()))} вакансий в оценке", flush=True)

def metrics(ranked):  # ranked: {title: [skill_idx или -1 (промах)]}
    P5, R10, P10, AP = [], [], [], []
    for g, lst in ranked.items():
        for i in by_title[g]:
            y = Y[i]; n = y.sum()
            rel = np.array([y[j] if j >= 0 else 0 for j in lst[:K_LLM]] + [0] * max(0, K_LLM - len(lst)))
            P5.append(rel[:5].mean()); P10.append(rel[:10].mean()); R10.append(rel[:10].sum() / n)
            AP.append(float((np.cumsum(rel) / np.arange(1, K_LLM + 1) * rel).sum() / min(n, K_LLM)))
    return {"P@5": round(float(np.mean(P5)), 4), "P@10": round(float(np.mean(P10)), 4),
            "R@10": round(float(np.mean(R10)), 4), "MAP@20": round(float(np.mean(AP)), 4)}

def to_vocab(skill):
    s = norm(skill)
    if s in idx: return idx[s]
    m = process.extractOne(s, vocab, scorer=fuzz.ratio, score_cutoff=92)
    return idx[m[0]] if m else -1

results = {}
# rubert-tiny2 (опубликованная модель)
rec = SkillRecommender(os.environ.get("TINY_PATH", "MrAlexGov/it-job-title-to-skills-rubert-tiny2"))
results["rubert-tiny2 (наша, 29M)"] = metrics({g: [idx[s] for s, _ in rec.recommend(orig[g], top_k=K_LLM, min_prob=0)] for g in sample})
# kNN по TF-IDF названий
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
Xtr = vec.fit_transform([titles[i] for i in tr]); nn = NearestNeighbors(n_neighbors=50, metric="cosine").fit(Xtr)
d, nb = nn.kneighbors(vec.transform([orig[g] for g in sample])); pop = Y[tr].mean(0)
sc = np.einsum("nk,nkl->nl", 1 - d, Y[tr][nb]) + 1e-3 * pop
results["TF-IDF kNN (бейзлайн)"] = metrics({g: list(np.argsort(-sc[n])[:K_LLM]) for n, g in enumerate(sample)})
print(json.dumps(results, ensure_ascii=False), flush=True)

EXAMPLE = ", ".join(vocab[:40])
PROMPT = """Ты эксперт по рынку труда в IT (Россия, hh.ru). Для вакансии с названием «{title}» перечисли {k} ключевых навыков,
которые работодатель скорее всего укажет в поле «Ключевые навыки» на hh.ru, от самого вероятного к менее вероятному.
Пиши названия навыков коротко, как на hh.ru (примеры стиля: {ex}). Только профессиональные навыки, без soft skills.
Ответ — только JSON-массив строк, без пояснений."""

def ask(model, g):
    for attempt in range(8):
        try:
            txt = kbench.llms[model].prompt(PROMPT.format(title=orig[g], k=K_LLM, ex=EXAMPLE))
            txt = txt if isinstance(txt, str) else str(txt)
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)   # рассуждения reasoning-моделей
            start = txt.find("[")
            return json.JSONDecoder().raw_decode(txt[start:])[0] if start >= 0 else []
        except Exception as e:
            err = repr(e); time.sleep(min(60, 5 * 2 ** attempt))   # 429 «heavy load» — ждём дольше
    return {"error": err}

models = [m for m in os.environ.get("LLMS_AVAILABLE", "").split(",") if m] if len(sys.argv) < 2 else sys.argv[1].split(",")
raw = json.load(open("llm_raw.json")) if os.path.exists("llm_raw.json") else {}
for model in models:
    raw.setdefault(model, {})
    todo = [g for g in sample if g not in raw[model] or isinstance(raw[model][g], dict)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", 8))) as ex:
        for g, ans in zip(todo, ex.map(lambda g: ask(model, g), todo)): raw[model][g] = ans
    json.dump(raw, open("llm_raw.json", "w"), ensure_ascii=False, indent=1)
    ok = {g: a for g, a in raw[model].items() if g in set(sample) and isinstance(a, list) and a}
    mapped = {g: [to_vocab(s) for s in a if isinstance(s, str)][:K_LLM] for g, a in ok.items()}
    flat = [j for l in mapped.values() for j in l]
    res = metrics({g: mapped.get(g, []) for g in sample})
    res.update({"answered": f"{len(ok)}/{len(sample)}", "mapped_to_vocab": round(sum(j >= 0 for j in flat) / max(len(flat), 1), 3),
                "sec": round(time.time() - t0)})
    results[model] = res
    print(model, json.dumps(res, ensure_ascii=False), flush=True)
json.dump(results, open("llm_vs_tiny.json", "w"), ensure_ascii=False, indent=2)
