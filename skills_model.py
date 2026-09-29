"""Инференс: название IT-должности → рекомендуемые навыки."""
import json
import os

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer


class SkillRecommender:
    def __init__(self, path="MrAlexGov/it-job-title-to-skills-rubert-tiny2", device="cpu"):
        if not os.path.isdir(path):
            path = snapshot_download(path)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.enc = AutoModel.from_pretrained(path).to(device).eval()
        self.vocab = json.load(open(os.path.join(path, "skill_vocab.json"), encoding="utf-8"))
        self.head = torch.nn.Linear(self.enc.config.hidden_size, len(self.vocab))
        self.head.load_state_dict(torch.load(os.path.join(path, "head.pt"), map_location=device))
        self.head.to(device).eval()
        self.device = device

    @torch.no_grad()
    def recommend(self, title, top_k=15, min_prob=0.05):
        x = self.tok([title], truncation=True, max_length=32, return_tensors="pt").to(self.device)
        h = self.enc(**x).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).float()
        p = torch.sigmoid(self.head((h * m).sum(1) / m.sum(1)))[0]
        top = torch.argsort(p, descending=True)[:top_k]
        return [(self.vocab[i], round(float(p[i]), 3)) for i in top if p[i] >= min_prob]


if __name__ == "__main__":
    import sys
    r = SkillRecommender(sys.argv[2] if len(sys.argv) > 2 else "MrAlexGov/it-job-title-to-skills-rubert-tiny2")
    for s, p in r.recommend(sys.argv[1] if len(sys.argv) > 1 else "Python-разработчик"):
        print(f"{p:.2f}  {s}")
