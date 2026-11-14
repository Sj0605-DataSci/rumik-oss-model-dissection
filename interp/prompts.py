"""Prompt grid shared by vllm_gen.py (stage A) and capture_acts.py (stage B)."""
import random

SPEAKERS = ["Ira", "Aisha", "Siya", "Zoya"]
TONES = ["happy", "sad", "angry", "excited", "professional"]
ACCENTS = ["Hindi accent", "Indian English"]
PACES = ["slow", "fast", "steady"]
TAGS = [None, None, "<laugh>", "<sigh>", "<chuckle>"]

TEXTS_HI = [
    "आज की meeting में सबको हमारा idea पसंद आया।",
    "मुझे नहीं पता था कि तुम इतनी जल्दी आ जाओगे।",
    "कल बारिश होगी, तो छाता ले जाना मत भूलना।",
    "ये खाना बहुत अच्छा बना है, थोड़ा और दो ना।",
    "तुमने फिर से वही गलती कर दी, मैं क्या करूँ?",
    "हमारी train दो घंटे late है, अब क्या करें।",
    "चलो आज शाम को कहीं घूमने चलते हैं।",
    "मेरा phone फिर से खो गया, यकीन नहीं होता।",
]
TEXTS_HI_V2 = TEXTS_HI + [
    "माँ ने कहा था कि जल्दी घर आना, लेकिन मैं भूल गया।",
    "इस बार की छुट्टियों में हम पहाड़ों पर जाएंगे।",
    "तुम्हारी आवाज़ सुनकर बहुत अच्छा लगा।",
    "ऑफिस में आज बहुत काम था, मैं थक गई हूँ।",
    "क्या तुमने वो नई film देखी? बहुत मज़ेदार है।",
    "डॉक्टर ने कहा है कि रोज़ सुबह टहलना ज़रूरी है।",
    "मैंने तुम्हारे लिए चाय बना दी है, ठंडी हो जाएगी।",
    "परीक्षा का result कल आने वाला है, डर लग रहा है।",
]
TEXTS_EN_V2 = None  # filled below
TEXTS_EN = [
    "I can't believe we actually finished the project on time.",
    "Please make sure the report is ready by tomorrow morning.",
    "Honestly, I thought the movie would be much better than that.",
    "We just got the news, and everyone is so happy about it.",
    "Your appointment has been rescheduled to Friday at three.",
    "I waited for an hour and nobody even called me back.",
    "Let me check that for you, it will only take a minute.",
    "That was the funniest thing I have heard all week.",
]
TEXTS_EN_V2 = TEXTS_EN + [
    "The weather has been so strange this whole month.",
    "Could you please send me the address once more?",
    "I finally learned how to make my grandmother's recipe.",
    "We missed the last bus, so we walked all the way home.",
    "Thank you so much for helping me with the move.",
    "The battery died right in the middle of the call.",
    "I have never seen the city this quiet before.",
    "Remember to water the plants while I am away.",
]


def build_prompts(n, seed, grid="v1"):
    """v1: accent tied to text language (confounded). v2: accent x language crossed, 16 texts/lang."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        accent = rng.choice(ACCENTS)
        if grid == "v2":
            lang = rng.choice(["hi", "en"])
            text = rng.choice(TEXTS_HI_V2 if lang == "hi" else TEXTS_EN_V2)
        else:
            lang = "hi" if accent == "Hindi accent" else "en"
            text = rng.choice(TEXTS_HI if lang == "hi" else TEXTS_EN)
        tag = rng.choice(TAGS)
        if tag:  # insert tag at a word boundary
            words = text.split()
            k = rng.randint(1, len(words) - 1)
            text = " ".join(words[:k] + [tag] + words[k:])
        row = dict(clip_id=i, speaker=SPEAKERS[i % len(SPEAKERS)], tone=rng.choice(TONES),
                   accent=accent, lang=lang, pace=rng.choice(PACES), tag=tag, text=text)
        row["input"] = f'<description="{row["tone"]}, {accent}, {row["pace"]} pace"> {text}'
        rows.append(row)
    return rows
