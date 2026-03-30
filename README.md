# 🌙 DreamSense — Dream Analysis & Prediction System

---

## Project Description

DreamSense is an AI-powered dream analysis and prediction system that takes a user's dream description as input and generates interpretations from three unique perspectives:

- 🔮 **Astrological** — based on zodiac sign and celestial symbolism
- 🕉️ **Spiritual** — based on sacred symbols and soul-level meanings
- 🧠 **Psychological** — based on Jungian psychology and emotional analysis

The system uses a **Machine Learning model (Random Forest + TF-IDF)** trained on a dataset of 500 dreams to predict:
- The **dream category** (Spiritual, Fear, Transformation, Career, etc.)
- The **dream mood** (Anxious, Peaceful, Joyful, etc.)
- **Probability breakdown** of top categories

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install flask scikit-learn pandas numpy
```

### Step 2 — Run the app
```bash
cd dream_app
python app.py
```

### Step 3 — Open in browser
```
http://localhost:5000
```

---

## Project Structure

```
dream_app/
├── app.py              ← Flask backend + ML model
├── Dream.csv           ← Training dataset (499 dream records)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── templates/
    └── index.html      ← Frontend UI
```

---

## Tech Stack

| Component       | Technology Used                  |
|-----------------|----------------------------------|
| Backend         | Python + Flask                   |
| ML Model        | Random Forest Classifier         |
| Text Processing | TF-IDF Vectorizer (scikit-learn) |
| Frontend        | HTML5, CSS3, Vanilla JavaScript  |
| Dataset         | Dream.csv (499 records)          |

---

## Dataset Features Used
- `dream_description` → TF-IDF vectorized for ML prediction
- `dream_category` → Target variable (11 classes)
- `dream_mood` → Secondary target variable (16 moods)
- `zodiac_sign` → Used for astrological interpretation
- `symbols_identified` → Symbol-based multi-perspective analysis
- `age` → Used for psychological age-factor analysis

---

## Submitted By
Anshika 
Department: School of Computing
Academic Year: 2025–26
