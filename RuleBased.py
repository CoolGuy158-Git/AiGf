import os
from difflib import SequenceMatcher

class RuleEngine:
    def __init__(self, file="training_data.txt"):
        self.file = file
        self.data = []
        self.load_data()

    def load_data(self):
        if os.path.exists(self.file):
            with open(self.file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                i = 0
                while i < len(lines) - 1:
                    if lines[i].lower().startswith("you:") and lines[i+1].lower().startswith("ai:"):
                        prompt = lines[i][4:].strip()
                        response = lines[i+1][3:].strip()
                        self.data.append((prompt, response))
                        i += 2
                    else:
                        i += 1

    def similarity(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def get_response(self, user_input):
        if not user_input.strip():
            return "I can't respond now."

        best_match = None
        highest_score = 0
        for prompt, response in self.data:
            score = self.similarity(user_input, prompt)
            if score > highest_score:
                highest_score = score
                best_match = response
        if highest_score < 0.4:
            return user_input.strip()
        return best_match
