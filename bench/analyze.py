"""Итог бенчмарка по сохранённым ответам (llm_raw.json), без запросов к моделям.
strict  — несопоставленные со словарём навыки LLM остаются в списке как промахи;
lenient — несопоставленные выкидываются, список сдвигается (оценка «по существу», без штрафа за формулировки)."""
import json, os, sys
os.environ.setdefault("N_TITLES", "200")
sys.argv = ["x", "none"]
import importlib.util
spec = importlib.util.spec_from_file_location("b", "llm_vs_tiny.py"); src = open("llm_vs_tiny.py").read()
src = src.split("EXAMPLE = ")[0]                      # берём только данные, сплит, метрики и бейзлайны
ns = {}; exec(compile(src, "llm_vs_tiny.py", "exec"), ns)
metrics, to_vocab, sample, results = ns["metrics"], ns["to_vocab"], ns["sample"], ns["results"]
raw = json.load(open("llm_raw.json"))
out = {k: {"strict": v, "lenient": v} for k, v in results.items()}
for model, d in raw.items():
    ok = {g: a for g, a in d.items() if g in set(sample) and isinstance(a, list) and a}
    if len(ok) < 0.9 * len(sample):
        out[model] = {"note": f"ответов {len(ok)}/{len(sample)} — прокси перегружен, не оцениваем"}; continue
    mapped = {g: [to_vocab(s) for s in a if isinstance(s, str)][:20] for g, a in ok.items()}
    flat = [j for l in mapped.values() for j in l]
    out[model] = {"strict": metrics({g: mapped.get(g, []) for g in sample}),
                  "lenient": metrics({g: [j for j in mapped.get(g, []) if j >= 0] for g in sample}),
                  "answered": f"{len(ok)}/{len(sample)}", "mapped_to_vocab": round(sum(j >= 0 for j in flat) / len(flat), 3)}
json.dump(out, open("final_results.json", "w"), ensure_ascii=False, indent=2)
print(f"{'модель':42} {'MAP20 strict':>12} {'MAP20 lenient':>13} {'P@5 lenient':>11} {'в словаре':>9}")
for m, v in sorted(out.items(), key=lambda kv: -kv[1].get("lenient", {}).get("MAP@20", -1)):
    if "note" in v: print(f"{m:42} {v['note']}"); continue
    print(f"{m:42} {v['strict']['MAP@20']:>12} {v['lenient']['MAP@20']:>13} {v['lenient']['P@5']:>11} {v.get('mapped_to_vocab', 1):>9}")
