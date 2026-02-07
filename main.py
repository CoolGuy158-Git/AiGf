import os
import json
import customtkinter as ctk
from Expander import AiGF, expand_text, load_pairs, DEVICE, MODEL_FILE, train, ChatDataset, used_responses
from RuleBased import RuleEngine
import torch
import ctypes

SETTINGS_FILE = "settings.json"
default_settings = {
    "temperature": 0.6,
    "min_chars": 20,
    "max_chars": 30,
    "appearance": "Dark",
    "transparency": 1.0,
    "training_data": "",
}

if os.path.exists(SETTINGS_FILE):
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        settings = json.load(f)
else:
    settings = default_settings.copy()

pairs = load_pairs()
words = set()
for p, r in pairs:
    words.update(p.split())
    words.update(r.split())
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}

model = AiGF(len(word2idx)).to(DEVICE)
if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))

rule_engine = RuleEngine("training_data.txt")

def chat_window():
    chat_root = ctk.CTk()
    chat_root.title("AI GF Chat")
    chat_root.geometry("1000x600")
    chat_root.attributes("-alpha", settings.get("transparency", 1.0))

    main_frame = ctk.CTkFrame(chat_root)
    main_frame.pack(fill="both", expand=True)

    chat_frame = ctk.CTkFrame(main_frame)
    chat_frame.pack(side="left", fill="both", expand=True)

    chat_display = ctk.CTkTextbox(chat_frame, font=("Arial", 20))
    chat_display.pack(fill="both", expand=True)
    chat_display.configure(state="disabled")

    input_frame = ctk.CTkFrame(chat_root)
    input_frame.pack(side="bottom", fill="x")

    user_input = ctk.CTkEntry(input_frame)
    user_input.pack(side="left", fill="x", expand=True, padx=10, pady=5)

    send_btn = ctk.CTkButton(input_frame, text="Send")
    send_btn.pack(side="left", padx=5)

    sidebar_width = 300
    sidebar_frame = ctk.CTkFrame(main_frame, width=sidebar_width)
    sidebar_frame.pack(side="right", fill="y")
    sidebar_frame.pack_forget()

    def toggle_sidebar():
        if sidebar_frame.winfo_ismapped():
            sidebar_frame.pack_forget()
        else:
            sidebar_frame.pack(side="right", fill="y")

    toggle_btn = ctk.CTkButton(chat_frame, text="⚙️", width=30, command=toggle_sidebar)
    toggle_btn.pack(side="right", padx=5)

    ctk.CTkLabel(sidebar_frame, text="Settings", font=("Arial", 16)).pack(pady=5)

    ctk.CTkLabel(sidebar_frame, text="Temperature").pack()
    temp_slider = ctk.CTkSlider(sidebar_frame, from_=0.1, to=1.0, number_of_steps=90)
    temp_slider.set(settings.get("temperature", 0.6))
    temp_slider.pack(pady=5, padx=10)

    ctk.CTkLabel(sidebar_frame, text="Min Chars").pack()
    min_slider = ctk.CTkSlider(sidebar_frame, from_=5, to=50, number_of_steps=45)
    min_slider.set(settings.get("min_chars", 20))
    min_slider.pack(pady=5, padx=10)

    ctk.CTkLabel(sidebar_frame, text="Max Chars").pack()
    max_slider = ctk.CTkSlider(sidebar_frame, from_=20, to=100, number_of_steps=80)
    max_slider.set(settings.get("max_chars", 30))
    max_slider.pack(pady=5, padx=10)

    dark_btn = ctk.CTkButton(
        sidebar_frame,
        text="Toggle Dark/Light Mode",
        command=lambda: ctk.set_appearance_mode(
            "Dark" if ctk.get_appearance_mode() == "Light" else "Light"
        ),
    )
    dark_btn.pack(pady=5, padx=10, fill="x")

    ctk.CTkLabel(sidebar_frame, text="Transparency").pack()
    def update_alpha(val):
        chat_root.attributes("-alpha", float(val))
    trans_slider = ctk.CTkSlider(sidebar_frame, from_=0.1, to=1.0, number_of_steps=90, command=update_alpha)
    trans_slider.set(settings.get("transparency", 1.0))
    trans_slider.pack(pady=5, padx=10)

    ctk.CTkLabel(sidebar_frame, text="Training Data").pack(pady=5)
    training_text = ctk.CTkTextbox(sidebar_frame, height=100)
    training_text.pack(pady=5, padx=5, fill="x")
    if os.path.exists("training_data.txt"):
        with open("training_data.txt", "r", encoding="utf-8") as f:
            training_text.insert("end", f.read())

    def clear_used_responses():
        global used_responses
        used_responses.clear()
        if os.path.exists("already_used_responses.txt"):
            os.remove("already_used_responses.txt")
        ctypes.windll.user32.MessageBoxW(0, "Used responses cleared!", "Success", 0x40)

    ctk.CTkButton(sidebar_frame, text="Clear Used Responses", command=clear_used_responses).pack(pady=5, fill="x")

    def save_settings():
        settings["temperature"] = temp_slider.get()
        settings["min_chars"] = int(min_slider.get())
        settings["max_chars"] = int(max_slider.get())
        settings["appearance"] = ctk.get_appearance_mode()
        settings["transparency"] = trans_slider.get()
        settings["training_data"] = training_text.get("1.0", "end").strip()
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        ctypes.windll.user32.MessageBoxW(0, "Settings Saved!", "Success", 0x40)

    save_btn = ctk.CTkButton(sidebar_frame, text="Save Settings", command=save_settings)
    save_btn.pack(pady=10, fill="x", padx=5)

    def send_message():
        text = user_input.get()
        if not text.strip():
            return
        chat_display.configure(state="normal")
        chat_display.insert("end", f"You: {text}\n")
        chat_display.yview("end")
        user_input.delete(0, "end")
        base_seed = rule_engine.get_response(text)
        response = expand_text(model, base_seed, word2idx, idx2word, settings=settings)
        chat_display.insert("end", f"AI GF: {response}\n")
        chat_display.yview("end")
        chat_display.configure(state="disabled")

    send_btn.configure(command=send_message)
    chat_root.mainloop()

ctk.set_appearance_mode(settings.get("appearance", "Dark"))
ctk.set_default_color_theme("blue")

mode = input("Choose mode (train/chat): ").lower()

if mode == "train":
    dataset = ChatDataset(pairs, word2idx)
    train(model, total_epochs=55, word2idx=word2idx, local_dataset=dataset, idx2word=idx2word)
elif mode == "chat":
    chat_window()
