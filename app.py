from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle, os, json, re

app = Flask(__name__)

# ─── SYMBOL MEANINGS DATABASE ─────────────────────────────────────────────────
SYMBOL_MEANINGS = {
    "water":     {"astro": "Moon governs water — emotional tides and intuition are strongly activated",
                  "spiritual": "Water represents purification and the flow of divine energy through your soul",
                  "psych": "Water often mirrors your unconscious mind and suppressed emotions"},
    "snake":     {"astro": "Snake is ruled by Scorpio & Pluto — expect deep transformation and rebirth",
                  "spiritual": "Snake is the Kundalini energy rising — spiritual awakening is near",
                  "psych": "Snake symbolises hidden fears, sexuality, or a threatening person in life"},
    "flying":    {"astro": "Jupiter and Sagittarius influence — desire for freedom and higher wisdom",
                  "spiritual": "Flying represents the soul's liberation from earthly attachments",
                  "psych": "Flying reflects ambition, a desire to rise above problems, or escapism"},
    "teeth":     {"astro": "Saturn rules bones and structure — anxiety about your foundation in life",
                  "spiritual": "Teeth falling out is a warning to re-examine the words you speak",
                  "psych": "Classic anxiety dream — stress about appearance, control, or powerlessness"},
    "fire":      {"astro": "Sun, Mars, and Aries — passion, willpower, and a major life change",
                  "spiritual": "Fire is the divine spark — purification and enlightenment are ahead",
                  "psych": "Fire can signal burning ambition, anger, or a situation getting 'out of hand'"},
    "forest":    {"astro": "Venus and Taurus — a need to reconnect with nature and your roots",
                  "spiritual": "Forests represent the unconscious sacred — spirits and guides dwell here",
                  "psych": "Getting lost in a forest reflects confusion about your life direction"},
    "ocean":     {"astro": "Neptune and Pisces — deep intuition, illusion, and spiritual sensitivity",
                  "spiritual": "The ocean is the cosmic womb — the source of all life and consciousness",
                  "psych": "The ocean is the collective unconscious — Jung saw it as the self beyond ego"},
    "temple":    {"astro": "Jupiter and Sagittarius — a spiritual quest and search for higher meaning",
                  "spiritual": "Temple visions indicate contact with a higher power or divine calling",
                  "psych": "Temples represent your internal belief system and moral framework"},
    "moon":      {"astro": "The Moon governs dreams directly — your emotional self is sending messages",
                  "spiritual": "Moon in dreams signals feminine divine energy and psychic messages",
                  "psych": "The moon represents the unconscious, hidden emotions, and cyclical change"},
    "sun":       {"astro": "Sun in Aries or Leo placement — identity, ego, and vitality are highlighted",
                  "spiritual": "The sun is the cosmic consciousness — divine illumination is entering your life",
                  "psych": "Sun symbolizes the conscious self, confidence, and desire for recognition"},
    "running":   {"astro": "Mercury and Gemini — restless energy, a need to slow down and reflect",
                  "spiritual": "Running away signifies karmic baggage that must be faced in this lifetime",
                  "psych": "Running from something indicates avoidance of a real-life problem or fear"},
    "death":     {"astro": "Pluto and Scorpio — a profound ending that makes space for a new beginning",
                  "spiritual": "Death dreams are rarely literal — they herald spiritual rebirth and transformation",
                  "psych": "Death often represents a desired end to a phase, relationship, or old identity"},
    "mirror":    {"astro": "Venus rules mirrors — questions of self-worth and how others perceive you",
                  "spiritual": "Mirrors show the soul — distorted reflections warn of self-deception",
                  "psych": "Mirrors reflect self-image, identity, and the 'shadow self' (Jung)"},
    "clock":     {"astro": "Saturn rules time — deadlines, discipline, and fear of running out of time",
                  "spiritual": "Clock dreams urge you to recognise the transience of material existence",
                  "psych": "Clocks signal pressure, anxiety about time, and unresolved urgency"},
    "bird":      {"astro": "Mercury and air signs — messages, communication, and swift movement",
                  "spiritual": "Birds carry messages from ancestors and spirit guides",
                  "psych": "Birds represent freedom, goals, and the human desire to transcend limits"},
    "lotus":     {"astro": "Venus and Neptune — beauty emerging from suffering, spiritual blossoming",
                  "spiritual": "Lotus is the highest spiritual symbol — enlightenment rising from darkness",
                  "psych": "Lotus reflects resilience — the ability to grow and thrive despite hardship"},
    "elephant":  {"astro": "Jupiter — abundance, good fortune, and powerful wisdom arriving",
                  "spiritual": "Elephant is Lord Ganesha — obstacles are being removed from your path",
                  "psych": "Elephants represent memory, wisdom, and confronting something long avoided"},
    "phoenix":   {"astro": "Scorpio and Pluto — complete destruction followed by glorious renewal",
                  "spiritual": "Phoenix is the ultimate symbol of spiritual death and rebirth",
                  "psych": "Phoenix dreams come when you are rebuilding your identity after a crisis"},
    "home":      {"astro": "Cancer and Moon — family security, roots, and emotional needs",
                  "spiritual": "Home represents the soul's true resting place and ancestral connections",
                  "psych": "Dreaming of home reflects a longing for safety, comfort, or the past"},
    "mountain":  {"astro": "Capricorn and Saturn — hard work, ambition, and a long climb to success",
                  "spiritual": "Mountains are sacred spaces — divine vision is available at the summit",
                  "psych": "Mountains represent challenges ahead — the dream asks: are you willing to climb?"},
}

ZODIAC_ASTRO = {
    "Aries":       "Mars energizes your dream life — bold visions and warrior energy dominate. Expect dreams of beginnings.",
    "Taurus":      "Venus paints your dreams with beauty and stability. Sensory, earthy dreams reflect your need for security.",
    "Gemini":      "Mercury fills your dreams with messages, duality, and rapid scene changes. Pay attention to what is spoken.",
    "Cancer":      "The Moon rules you — your dreams are especially vivid, emotional, and tied to family and past memories.",
    "Leo":         "The Sun illuminates your dreams. Expect grand themes, creative visions, and a spotlight on your identity.",
    "Virgo":       "Mercury brings analytical dreams filled with detail, order, and practical problem-solving scenarios.",
    "Libra":       "Venus creates dreams of beauty, relationships, and balance. Recurring themes of fairness and harmony.",
    "Scorpio":     "Pluto dives deep — your dreams are intense, transformative, and full of symbolic death and rebirth cycles.",
    "Sagittarius": "Jupiter expands your dream world — adventure, philosophy, and far-off lands fill your nightly visions.",
    "Capricorn":   "Saturn structures your dreams around discipline, responsibility, and long-term ambitions.",
    "Aquarius":    "Uranus brings unusual, inventive, even prophetic dreams. Your subconscious thinks ahead of its time.",
    "Pisces":      "Neptune rules you — you are the most psychic dreamer. Your dreams blur reality and contain deep messages.",
}

CATEGORY_SPIRITUAL = {
    "Spiritual":      "This dream is a direct message from your higher self or a divine source. Meditate on its imagery.",
    "Transformation": "Your soul is undergoing a profound metamorphosis. Old karmic cycles are breaking and new ones forming.",
    "Fear":           "Fear dreams are sacred teachers. The universe is showing you where your spiritual blocks reside.",
    "Ambition":       "Your soul is pushing you toward your dharma (life purpose). Trust the urge to grow and achieve.",
    "Career":         "Material pursuits are part of your karsha (cosmic duty). This dream aligns your work with your soul path.",
    "Adventure":      "Your spirit is restless and seeking new experiences to evolve. A journey — inner or outer — is near.",
    "Stress":         "This dream is a call to surrender. The universe asks you to release control and trust divine timing.",
    "Relationships":  "Karmic connections are at play. Souls from past lives are reflected in the people of this dream.",
    "Creativity":     "Creative inspiration is flowing from a higher source. You are being called to create something meaningful.",
    "Family":         "Ancestral energy surrounds this dream. Your lineage is speaking — listen for inherited patterns.",
    "Reflective":     "Your soul is pausing to review your journey. This is a dream of spiritual self-assessment.",
}

CATEGORY_PSYCH = {
    "Spiritual":      "You may be searching for meaning beyond the material. This dream reflects existential questioning.",
    "Transformation": "Psychologically, you are in the individuation process (Jung) — integrating shadow aspects of self.",
    "Fear":           "Your amygdala is processing unresolved fears. These dreams are your brain rehearsing how to cope.",
    "Ambition":       "Your ego is striving for achievement. The dream reflects drive, competitive instincts, and self-worth.",
    "Career":         "Work-related anxiety or aspiration is surfacing. Your subconscious is problem-solving your career path.",
    "Adventure":      "A need for novelty and stimulation is driving this dream. You may feel constrained in waking life.",
    "Stress":         "Classic stress response — cortisol affects dream content. Your brain is trying to regulate emotions.",
    "Relationships":  "Attachment patterns, projection, and unspoken feelings are being processed through this dream.",
    "Creativity":     "Your right brain is highly active. Creative blocks may be dissolving as new neural paths form.",
    "Family":         "Family systems theory: unresolved dynamics, roles, and loyalties are being re-examined in sleep.",
    "Reflective":     "Your brain is in consolidation mode — integrating recent experiences into long-term memory.",
}

MOOD_PSYCH = {
    "Anxious":    "Anxiety in dreams reflects your nervous system in a hypervigilant state. Consider stress management.",
    "Peaceful":   "Peaceful dreams indicate emotional regulation and a sense of inner security are well-balanced.",
    "Exciting":   "Excitement signals dopamine activity — your brain is anticipating reward or positive change.",
    "Fearful":    "Fear in dreams is the amygdala processing threat responses. Examine what feels unsafe in waking life.",
    "Joyful":     "Joy in dreams correlates with positive well-being. Your emotional life is in a healthy state.",
    "Sad":        "Sadness is often suppressed grief or loss being processed safely during sleep.",
    "Frustrated": "Frustration dreams point to blocked goals or communication difficulties in daily life.",
    "Wonder":     "A sense of wonder reflects an open, curious mind and strong imaginative capacity.",
    "Determined": "This mood signals high executive function — your prefrontal cortex is goal-oriented even in sleep.",
    "Inspired":   "Inspiration dreams occur when the subconscious has connected seemingly unrelated ideas.",
    "Confused":   "Confusion may reflect information overload or an unresolved decision you are trying to make.",
    "Brave":      "Bravery in dreams shows psychological growth — you are facing and overcoming your inner obstacles.",
    "Confident":  "Confidence signals a strong sense of self-efficacy and positive self-concept.",
    "Empowered":  "Empowerment dreams come when you are reclaiming personal agency after a period of powerlessness.",
    "Creative":   "Creative mood activates the default mode network — your brain's imagination hub is fully online.",
    "Reflective": "Reflective mood indicates metacognitive activity — your mind is examining itself from a distance.",
}


# ─── TRAIN MODEL ──────────────────────────────────────────────────────────────
def train_model():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Dream.csv'), on_bad_lines='skip')
    df = df.dropna(subset=['dream_description', 'dream_category', 'dream_mood'])

    le_cat  = LabelEncoder()
    le_mood = LabelEncoder()
    df['cat_enc']  = le_cat.fit_transform(df['dream_category'])
    df['mood_enc'] = le_mood.fit_transform(df['dream_mood'])

    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    X = tfidf.fit_transform(df['dream_description']).toarray()

    clf_cat  = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_mood = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_cat.fit(X, df['cat_enc'])
    clf_mood.fit(X, df['mood_enc'])

    return tfidf, clf_cat, clf_mood, le_cat, le_mood

TFIDF, CLF_CAT, CLF_MOOD, LE_CAT, LE_MOOD = train_model()


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def extract_symbols(text):
    text_lower = text.lower()
    found = []
    for sym in SYMBOL_MEANINGS:
        if sym in text_lower:
            found.append(sym)
    return found[:5]  # top 5


def predict_dream(description, zodiac, age):
    X = TFIDF.transform([description]).toarray()
    cat_idx   = CLF_CAT.predict(X)[0]
    mood_idx  = CLF_MOOD.predict(X)[0]
    cat_probs = CLF_CAT.predict_proba(X)[0]
    mood_probs= CLF_MOOD.predict_proba(X)[0]

    category  = LE_CAT.inverse_transform([cat_idx])[0]
    mood      = LE_MOOD.inverse_transform([mood_idx])[0]

    # top 3 categories
    top3_cat_idx  = np.argsort(cat_probs)[::-1][:3]
    top3_cat      = [(LE_CAT.inverse_transform([i])[0], round(cat_probs[i]*100, 1)) for i in top3_cat_idx]

    symbols = extract_symbols(description)

    # Build interpretations
    astro_parts = []
    astro_parts.append(f" <b>Zodiac Influence ({zodiac}):</b> {ZODIAC_ASTRO.get(zodiac, 'Your zodiac carries unique cosmic energy in this dream.')}")
    if symbols:
        sym_lines = []
        for s in symbols:
            m = SYMBOL_MEANINGS.get(s, {})
            if m:
                sym_lines.append(f"<b>{s.capitalize()}:</b> {m['astro']}")
        if sym_lines:
            astro_parts.append(" <b>Planetary Symbol Readings:</b><br>" + "<br>".join(sym_lines))
    astro_parts.append(f" <b>Dream Category Forecast ({category}):</b> This dream type aligns with celestial patterns. Trust the timing — the cosmos rarely sends messages by accident.")

    spiritual_parts = []
    spiritual_parts.append(f"️ <b>Soul Message ({category}):</b> {CATEGORY_SPIRITUAL.get(category, 'Your higher self is speaking through this dream.')}")
    if symbols:
        sym_lines = []
        for s in symbols:
            m = SYMBOL_MEANINGS.get(s, {})
            if m:
                sym_lines.append(f"<b>{s.capitalize()}:</b> {m['spiritual']}")
        if sym_lines:
            spiritual_parts.append(" <b>Sacred Symbol Wisdom:</b><br>" + "<br>".join(sym_lines))
    spiritual_parts.append(f" <b>Spiritual Mood Reading ({mood}):</b> The emotional tone of this dream carries a clear spiritual vibration — one of awakening and growth.")

    psych_parts = []
    psych_parts.append(f" <b>Psychological Theme ({category}):</b> {CATEGORY_PSYCH.get(category, 'Your subconscious is processing important emotional material.')}")
    psych_parts.append(f" <b>Emotional State Analysis ({mood}):</b> {MOOD_PSYCH.get(mood, 'Your dream mood reveals key emotional patterns.')}")
    if symbols:
        sym_lines = []
        for s in symbols:
            m = SYMBOL_MEANINGS.get(s, {})
            if m:
                sym_lines.append(f"<b>{s.capitalize()}:</b> {m['psych']}")
        if sym_lines:
            psych_parts.append(" <b>Symbol Psychology:</b><br>" + "<br>".join(sym_lines))
    if int(age) < 25:
        psych_parts.append(" <b>Age Factor:</b> At your stage of development, identity-formation dreams are common — your psyche is still exploring who you are.")
    elif int(age) < 40:
        psych_parts.append(" <b>Age Factor:</b> Dreams in early adulthood often revolve around achievement, relationships, and establishing your place in the world.")
    else:
        psych_parts.append(" <b>Age Factor:</b> Midlife dreams frequently explore meaning, legacy, and the integration of life experience — a Jungian individuation process.")

    return {
        "category": category,
        "mood": mood,
        "top3": top3_cat,
        "symbols": symbols,
        "astro": "<br><br>".join(astro_parts),
        "spiritual": "<br><br>".join(spiritual_parts),
        "psych": "<br><br>".join(psych_parts),
        "confidence": round(float(cat_probs[cat_idx]) * 100, 1)
    }


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    description = data.get('description', '').strip()
    zodiac = data.get('zodiac', 'Aries')
    age = data.get('age', 25)

    if len(description) < 20:
        return jsonify({"error": "Please describe your dream in more detail (at least 20 characters)."})

    result = predict_dream(description, zodiac, age)
    return jsonify(result)


if __name__ == '__main__':
    print(" Dream Analysis System starting...")
    print("   Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)
