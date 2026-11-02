"""Long-context tasks: needle-in-a-haystack and a LongBench subset.

Both return items: {"id", "prompt_ids", "max_new", "score": fn(pred_text) -> float, ...}
"""
import json
import os
import random
import re
import string
from collections import Counter
from difflib import SequenceMatcher

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "longbench_raw", "data")


def _read(task, e=True):
    path = os.path.join(DATA, f"{task}{'_e' if e else ''}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True)


# ----------------------------------------------------------------------------- needle in a haystack
KEYS = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tbilisi", "Hanoi", "Oslo", "Lima", "Perth", "Accra"]


def haystack_text():
    """Real English prose (US government reports from LongBench), not repeated filler:
    repetitive haystacks are trivially compressible and make every method look perfect."""
    return "\n\n".join(r["context"] for r in _read("gov_report"))


def niah_items(tok, lengths=(2048, 4096, 8192, 16384), depths=(0.0, 0.25, 0.5, 0.75, 1.0), seed=0):
    rng = random.Random(seed)
    hay = tok.encode(haystack_text())
    items = []
    for L in lengths:
        for d in depths:
            key, num = rng.choice(KEYS), rng.randint(1_000_000, 9_999_999)
            needle = f" The special magic number for {key} is: {num}. "
            q = (f"\n\nWhat is the special magic number for {key} mentioned in the text above? "
                 f"Answer with the number only.")
            head = "Read the following text carefully. Somewhere in it is a special magic number.\n\n"
            overhead = len(tok.encode(chat(tok, head + needle + q)))
            start = rng.randint(0, len(hay) - L - 1)
            ctx = hay[start:start + L - overhead]
            # insert at the sentence end nearest *before* the requested depth
            j = int(len(ctx) * d)
            if 0 < j < len(ctx):
                for k in range(j, max(0, j - 300), -1):
                    if tok.decode(ctx[k - 1:k]).rstrip().endswith("."):
                        j = k
                        break
            before, after = tok.decode(ctx[:j]), tok.decode(ctx[j:])
            prompt = chat(tok, head + before + needle + after + q)
            ids = tok.encode(prompt)
            items.append({"id": f"niah-{L}-{d}", "task": "niah", "length": L, "depth": d,
                          "prompt_ids": ids, "max_new": 12, "answer": str(num),
                          "score": (lambda ans: lambda p: float(ans in re.sub(r"[ ,]", "", p)))(str(num))})
    return items


# ----------------------------------------------------------------------------- LongBench
# prompt formats and generation lengths from LongBench's config (dataset2prompt / dataset2maxlen)
PROMPTS = {
    "multifieldqa_en": ("Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following "
                        "question based on the above text, only give me the answer and do not output any other "
                        "words.\n\nQuestion: {input}\nAnswer:", 64),
    "hotpotqa": ("Answer the question based on the given passages. Only give me the answer and do not output any "
                 "other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the "
                 "given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}"
                 "\nAnswer:", 32),
    "2wikimqa": ("Answer the question based on the given passages. Only give me the answer and do not output any "
                 "other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the "
                 "given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}"
                 "\nAnswer:", 32),
    "triviaqa": ("Answer the question based on the given passage. Only give me the answer and do not output any "
                 "other words. The following are some examples.\n\n{context}\n\n{input}", 32),
    "trec": ("Please determine the type of the question below. Here are some examples of questions.\n\n"
             "{context}\n{input}", 64),
    "lcc": ("Please complete the code given below. \n{context}Next line of code:\n", 64),
}
NO_CHAT = {"triviaqa", "trec", "lcc"}     # LongBench feeds few-shot / code tasks without the chat template


def _normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, gt):
    p, g = _normalize(pred).split(), _normalize(gt).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def classification(pred, gt, all_classes):
    hits = [c for c in all_classes if c in pred]
    hits = [h for h in hits if not (h in gt and h != gt)]
    return (1.0 / len(hits)) if gt in hits else 0.0


def code_sim(pred, gt):
    # LongBench uses fuzzywuzzy.ratio; difflib's ratio is the same Ratcliff/Obershelp idea
    line = next((l for l in pred.lstrip("\n").split("\n")
                 if "`" not in l and "#" not in l and "//" not in l), "")
    return SequenceMatcher(None, line, gt).ratio()


def _scorer(task, answers, all_classes):
    def score(pred):
        if task in ("triviaqa", "trec"):
            pred = pred.lstrip("\n").split("\n")[0]
        best = 0.0
        for gt in answers:
            if task == "trec":
                s = classification(pred, gt, all_classes)
            elif task == "lcc":
                s = code_sim(pred, gt)
            else:
                s = qa_f1(pred, gt)
            best = max(best, s)
        return best
    return score


def longbench_items(tok, tasks=("multifieldqa_en", "hotpotqa", "2wikimqa", "triviaqa", "trec", "lcc"),
                    n_per_task=20, max_tokens=8192, seed=0):
    """LongBench-E samples from the 4k-8k word bucket, middle-truncated to max_tokens
    (LongBench's own truncation rule)."""
    rng = random.Random(seed)
    items = []
    for t in tasks:
        rows = [r for r in _read(t) if 4000 <= r["length"] < 8000]
        rng.shuffle(rows)
        fmt, max_new = PROMPTS[t]
        for r in rows[:n_per_task]:
            text = fmt.format(context=r["context"], input=r["input"])
            ids = tok.encode(text)
            if len(ids) > max_tokens - 64:
                half = (max_tokens - 64) // 2
                text = tok.decode(ids[:half]) + tok.decode(ids[-half:])
            prompt = text if t in NO_CHAT else chat(tok, text)
            items.append({"id": f"{t}-{r['_id']}", "task": t, "prompt_ids": tok.encode(prompt),
                          "max_new": max_new, "answers": r["answers"],
                          "score": _scorer(t, r["answers"], r.get("all_classes") or [])})
    return items
