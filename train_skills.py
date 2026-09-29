"""Название IT-должности → ключевые навыки (multi-label). Данные: hh.ru 04–05.2023 (Kaggle jobs-raw-data, CC BY 4.0).
Сплит по уникальным названиям: в тесте только должности, которых модель не видела."""
import json, glob, time, numpy as np, torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda"
src = glob.glob("/kaggle/input/**/it_title_skills.jsonl", recursive=True)[0]
vocab = json.load(open(glob.glob("/kaggle/input/**/skill_vocab.json", recursive=True)[0]))
idx = {s: i for i, s in enumerate(vocab)}
rows = [json.loads(l) for l in open(src)]
titles = [r["title"] for r in rows]
Y = np.zeros((len(rows), len(vocab)), dtype=np.float32)
for i, r in enumerate(rows):
    for s in r["skills"]: Y[i, idx[s]] = 1
groups = [t.lower().strip() for t in titles]
gss = GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
tr, te = next(gss.split(titles, groups=groups))
a, b = next(GroupShuffleSplit(1, test_size=0.1, random_state=SEED).split(tr, groups=[groups[i] for i in tr]))
tr_idx, va_idx = tr[a], tr[b]
print(f"train {len(tr_idx)} val {len(va_idx)} test {len(te)}; навыков {len(vocab)}")

def evaluate(scores, Yt, ks=(5, 10)):
    out = {}
    order = np.argsort(-scores, 1)
    for k in ks:
        hit = np.take_along_axis(Yt, order[:, :k], 1)
        out[f"P@{k}"] = round(float(hit.mean()), 4)
        out[f"R@{k}"] = round(float((hit.sum(1) / np.maximum(Yt.sum(1), 1)).mean()), 4)
    # mean average precision по вакансии
    aps = []
    for s, y in zip(order[:, :50], Yt):
        rel = y[s]; n = y.sum()
        if n == 0: continue
        aps.append(float((np.cumsum(rel) / np.arange(1, 51) * rel).sum() / min(n, 50)))
    out["MAP@50"] = round(float(np.mean(aps)), 4)
    return out

report = {"n_train": int(len(tr_idx)), "n_test": int(len(te)), "n_skills": len(vocab)}
# 1) популярность
pop = Y[tr_idx].mean(0)
report["popularity"] = evaluate(np.tile(pop, (len(te), 1)), Y[te])
# 2) kNN по TF-IDF символьных n-грамм названия
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, lowercase=True)
Xtr = vec.fit_transform([titles[i] for i in tr_idx]); Xte = vec.transform([titles[i] for i in te])
best = None
for k in (10, 25, 50, 100):
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Xtr)
    Xva = vec.transform([titles[i] for i in va_idx]); d, nb = nn.kneighbors(Xva)
    sc = np.einsum("nk,nkl->nl", 1 - d, Y[tr_idx][nb]) + 1e-3 * pop
    m = evaluate(sc, Y[va_idx])["MAP@50"]
    if best is None or m > best[0]: best = (m, k)
nn = NearestNeighbors(n_neighbors=best[1], metric="cosine").fit(Xtr); d, nb = nn.kneighbors(Xte)
report["tfidf_knn"] = {"k": best[1], **evaluate(np.einsum("nk,nkl->nl", 1 - d, Y[tr_idx][nb]) + 1e-3 * pop, Y[te])}
print(json.dumps(report, ensure_ascii=False), flush=True)

# 3) дообучение BERT-энкодера с multi-label головой
class M(torch.nn.Module):
    def __init__(self, name):
        super().__init__(); self.enc = AutoModel.from_pretrained(name)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, len(vocab))
    def forward(self, **x):
        h = self.enc(**x).last_hidden_state; m = x["attention_mask"].unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1))

def train(name, epochs=8, lr=5e-5, bs=64):
    t0 = time.time(); tok = AutoTokenizer.from_pretrained(name); model = M(name).to(dev)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": lr}, {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    enc = lambda ts: tok(ts, truncation=True, max_length=32, padding=True, return_tensors="pt").to(dev)
    def predict(ids):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(ids), 256): out.append(torch.sigmoid(model(**enc([titles[j] for j in ids[i:i+256]]))).float().cpu().numpy())
        return np.concatenate(out)
    scaler = torch.cuda.amp.GradScaler(); best, state = -1, None
    Yt = torch.tensor(Y)
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i+bs]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(**enc([titles[j] for j in b])).float(), Yt[b].to(dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        m = evaluate(predict(va_idx), Y[va_idx])["MAP@50"]
        print(f"{name} epoch {ep+1}: val MAP@50 {m}", flush=True)
        if m > best: best, state, best_ep = m, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1
    model.load_state_dict(state)
    return {"best_epoch": best_ep, "train_min": round((time.time() - t0) / 60, 1), **evaluate(predict(te), Y[te])}, model, tok

for name in ["cointegrated/rubert-tiny2", "ai-forever/ruBert-base"]:
    res, model, tok = train(name)
    report[name] = res
    print(json.dumps({name: res}, ensure_ascii=False), flush=True)
    if name == "cointegrated/rubert-tiny2":
        model.enc.save_pretrained("model"); tok.save_pretrained("model"); torch.save(model.head.state_dict(), "model/head.pt")
        json.dump(vocab, open("model/skill_vocab.json", "w"), ensure_ascii=False)
        model.eval()
        for t in ["Python-разработчик", "DevOps инженер", "Системный аналитик", "Frontend-разработчик React", "Специалист по промпт-инжинирингу", "Разработчик 1С", "Data Scientist", "QA Automation Engineer", "Backend-разработчик Go", "Специалист технической поддержки"]:
            with torch.no_grad(): p = torch.sigmoid(model(**tok([t], return_tensors="pt").to(dev)))[0].cpu().numpy()
            report.setdefault("examples", {})[t] = [(vocab[i], round(float(p[i]), 2)) for i in np.argsort(-p)[:10]]
json.dump(report, open("metrics.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
