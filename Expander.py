import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import requests
import re
from bs4 import BeautifulSoup
import random
import language_tool_python
from collections import Counter

SEQ_LEN = 20
HIDDEN = 512
EMBEDDING = 256
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_FILE = "ai_gf.pt"
USED_FILE = "already_used_responses.txt"
tool = language_tool_python.LanguageTool('en-US')

NSFW_KEYWORDS = ["porn", "sex", "xxx", "nude", "nsfw", "hentai", "erotic"]

def is_safe_text(text):
    t = text.lower()
    return not any(k in t for k in NSFW_KEYWORDS)

class AiGF(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, EMBEDDING)
        self.lstm = nn.LSTM(EMBEDDING, HIDDEN, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(HIDDEN, vocab_size)

    def forward(self, x):
        x = self.token_emb(x)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.fc(out)

class ChatDataset(Dataset):
    def __init__(self, pairs, word2idx):
        self.word2idx = word2idx
        self.data = []
        for prompt, response in pairs:
            pt = [word2idx.get(w, 0) for w in prompt.strip().split()]
            rt = [word2idx.get(w, 0) for w in response.strip().split()]
            seq = pt + rt
            for i in range(len(seq) - SEQ_LEN):
                x = seq[i:i + SEQ_LEN]
                y = seq[i + 1:i + SEQ_LEN + 1]
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

def get_sources():
    return [
        "https://en.wiktionary.org/wiki/Special:Random",
        "https://www.dictionary.com/browse/example",
        "https://www.merriam-webster.com/dictionary/example",
        "https://www.collinsdictionary.com/dictionary/english/example",
        "https://www.oxfordlearnersdictionaries.com/definition/english/example",
        "https://www.macmillandictionary.com/dictionary/british/example",
        "https://www.ldoceonline.com/dictionary/example",
        "https://www.vocabulary.com/dictionary/example",
        "https://www.cambridge.org/dictionary/english/example",
        "https://www.yourdictionary.com/example",
        "https://www.thefreedictionary.com/example",
        "https://www.wordreference.com/definition/example",
        "https://www.infoplease.com/dictionary/example",
        "https://www.encyclopedia.com/dictionary/example",
        "https://www.ahdictionary.com/word/search.html?q=example",
        "https://www.reference.com/dictionary/example",
        "https://www.oxforddictionaries.com/definition/english/example",
        "https://en.wikipedia.org/wiki/Special:Random",
        "https://github.com/search?q=python",
        "https://www.reddit.com/r/python/",
        "https://www.bilibili.com/video/BV1xK4y1C7yZ",
        "https://www.englishclub.com/grammar/",
        "https://www.grammarly.com/blog/grammar-rules/",
        "https://www.ef.com/english-resources/english-grammar/",
        "https://www.perfect-english-grammar.com/",
        "https://www.usingenglish.com/guides/",
        "https://www.englishpage.com/grammar/",
        "https://www.englishgrammar.org/",
        "https://www.dailygrammar.com/",
        "https://learnenglish.britishcouncil.org/grammar"
    ]

def fetch_web_text(url, max_chars=2000):
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "table"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        if not is_safe_text(text):
            return None
        matches = tool.check(text)
        if matches:
            text = language_tool_python.utils.correct(text, matches)
        return text[:max_chars]
    except:
        return None

def build_web_pairs(max_pairs=500):
    sources = get_sources()
    random.shuffle(sources)
    pairs = []
    for site in sources:
        text = fetch_web_text(site, max_chars=2500)
        if not text: continue
        words = text.split()
        for i in range(0, len(words) - SEQ_LEN - 1, SEQ_LEN):
            prompt = " ".join(words[i:i + SEQ_LEN])
            response = " ".join(words[i + 1:i + SEQ_LEN + 1])
            pairs.append((prompt, response))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    return pairs

def train(model, total_epochs, word2idx, idx2word, local_dataset, web_pairs_per_epoch=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    local_pairs = [(x, y) for x, y in local_dataset.data]
    local_pairs = [( " ".join([idx2word.get(t, "<unk>") for t in x]),
                     " ".join([idx2word.get(t, "<unk>") for t in y])) for x, y in local_pairs]
    for epoch in range(total_epochs):
        web_pairs = build_web_pairs(max_pairs=web_pairs_per_epoch)
        all_pairs = local_pairs + web_pairs
        dataset = ChatDataset(all_pairs, word2idx)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{total_epochs} | Loss: {total_loss / len(loader):.4f} | Steps: {len(loader)}")

def load_pairs(file="training_data.txt"):
    pairs = []
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            i = 0
            while i < len(lines) - 1:
                if lines[i].lower().startswith("you:") and lines[i + 1].lower().startswith("ai:"):
                    prompt = lines[i][4:].strip()
                    response = lines[i + 1][3:].strip()
                    pairs.append((prompt, response))
                    i += 2
                else:
                    i += 1
    return pairs

if os.path.exists(USED_FILE):
    with open(USED_FILE, "r", encoding="utf-8") as f:
        used_responses = set(line.strip() for line in f if line.strip())
else:
    used_responses = set()

def save_used_response(resp):
    used_responses.add(resp)
    with open(USED_FILE, "a", encoding="utf-8") as f:
        f.write(resp + "\n")

def expand_text(
    model, prompt, word2idx, idx2word,
    min_chars=20, max_chars=50,
    top_k=100, temperature=0.6,
    attempts=20,
    settings=None
):
    global used_responses
    if settings is not None:
        min_chars = settings.get("min_chars", min_chars)
        max_chars = settings.get("max_chars", max_chars)
        temperature = settings.get("temperature", temperature)
    model.eval()
    with torch.no_grad():
        tokens = [word2idx.get(w, 0) for w in prompt.strip().split()]
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
        best_sentence = None
        best_score = float("inf")
        for _ in range(attempts):
            current_tokens = tokens.clone()
            generated = current_tokens[0].tolist()
            while True:
                logits = model(current_tokens)[:, -1, :] / temperature
                topk_logits, topk_indices = torch.topk(logits, top_k)
                probs = F.softmax(topk_logits, dim=-1)
                filtered_indices = [i for i in topk_indices[0] if generated.count(i.item()) < 2]
                if not filtered_indices:
                    filtered_indices = topk_indices[0]
                next_tok = random.choice(filtered_indices).view(1,1)
                current_tokens = torch.cat([current_tokens, next_tok], dim=1)
                generated.append(next_tok.item())
                word = idx2word.get(next_tok.item(), "")
                sentence_so_far = " ".join([idx2word.get(t, "<unk>") for t in generated])
                char_count = len(sentence_so_far)
                if word in ".!?" or char_count >= max_chars:
                    break
            words = [idx2word.get(t, "<unk>") for t in generated]
            sentence = " ".join(words).strip()
            char_count = len(sentence)
            word_counts = Counter(words)
            repeat_count = sum(c-2 for c in word_counts.values() if c >= 3)
            if char_count < min_chars:
                continue
            if sentence in used_responses:
                continue
            if repeat_count < best_score:
                best_score = repeat_count
                best_sentence = sentence
            if repeat_count == 0:
                break
        if best_sentence is None:
            best_sentence = sentence
        save_used_response(best_sentence)
        return best_sentence
