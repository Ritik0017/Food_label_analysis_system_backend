from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import pytesseract
import numpy as np
import pandas as pd
import os
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Windows only
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------
# PROFILE SCORE DATASET (XLSX)
# ---------------------------------------------------
PROFILE_XLSX_PATH = "health_scores_output.xlsx"

PROFILE_META = {
    "general": {"label": "General"},
    "diabetic": {"label": "Diabetic"},
    "heart_patient": {"label": "Heart Patient"},
    "athlete": {"label": "Athlete"},
    "child": {"label": "Child"},
    "elderly": {"label": "Elderly"},
    "weight_loss": {"label": "Weight Loss"},
    "vegan": {"label": "Vegan"},
}

PROFILE_SCORE_COLUMN_MAP = {
    "general": "score_general",
    "diabetic": "score_diabetic",
    "heart_patient": "score_heart_patient",
    "athlete": "score_athlete",
    "weight_loss": "score_weight_loss",
    # child / elderly / vegan are not present in current xlsx output by default
}

health_score_df = None

def load_health_score_dataset():
    global health_score_df
    if health_score_df is None:
        if os.path.exists(PROFILE_XLSX_PATH):
            health_score_df = pd.read_excel(PROFILE_XLSX_PATH)
            health_score_df.columns = [str(c).strip() for c in health_score_df.columns]
        else:
            health_score_df = pd.DataFrame()
    return health_score_df

# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not loaded properly.")

    # upscale for better OCR
    img = cv2.resize(img, None, fx=4.5, fy=4.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # contrast improve
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # sharpen
    kernel = np.array([
        [0, -1, 0],
        [-1, 5.2, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(gray, -1, kernel)

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12
    )

    return img, gray, thresh


# ---------------------------------------------------
# OCR HELPERS
# ---------------------------------------------------
def clean_joined_text(text):
    return " ".join(text.split()).strip()


def extract_text_with_best_config(images):
    configs = [
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 4',
        r'--oem 3 --psm 11',
        r'--oem 3 --psm 12'
    ]

    best_text = ""
    best_score = -1

    for img in images:
        for config in configs:
            text = pytesseract.image_to_string(img, config=config)
            cleaned = clean_joined_text(text)

            # score by text length + count of nutrition-like terms
            nutrition_terms = [
                "energy", "protein", "fat", "sugar", "sodium",
                "salt", "carbohydrate", "fiber", "saturated", "trans"
            ]
            hits = sum(1 for t in nutrition_terms if t in cleaned.lower())
            score = len(cleaned) + hits * 25

            if score > best_score:
                best_score = score
                best_text = cleaned

    return best_text


def extract_word_data(img):
    data = pytesseract.image_to_data(
        img,
        config='--oem 3 --psm 6',
        output_type=pytesseract.Output.DICT
    )

    words = []
    n = len(data["text"])

    for i in range(n):
        raw = data["text"][i].strip()
        if not raw:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1

        if conf < 0:
            continue

        left = data["left"][i]
        top = data["top"][i]
        width = data["width"][i]
        height = data["height"][i]

        words.append({
            "text": raw,
            "left": left,
            "top": top,
            "right": left + width,
            "bottom": top + height,
            "width": width,
            "height": height,
            "line_num": data["line_num"][i],
            "block_num": data["block_num"][i],
            "par_num": data["par_num"][i],
        })

    return words


# ---------------------------------------------------
# MAIN OCR
# ---------------------------------------------------
def extract_text_from_image(image_path):
    color, gray, thresh = preprocess_image(image_path)

    full_text = extract_text_with_best_config([thresh, gray])

    return full_text, color, gray, thresh


# ---------------------------------------------------
# INGREDIENT EXTRACTION
# KEEP THIS PART SAFE
# ---------------------------------------------------
def extract_ingredients_section(text):
    clean_text = " ".join(text.split())

    start_keywords = [
        "ingredients",
        "ingredient",
        "ingredlents",
        "ingrediants",
        "ingrdients",
        "ingrédients"
    ]

    end_keywords = [
        "contains",
        "allergen",
        "nutrition",
        "nutrition facts",
        "storage",
        "manufacturer",
        "fssai",
        "warning",
        "directions",
        "serving"
    ]

    start_pattern = r'(' + '|'.join(start_keywords) + r')[:\-]?\s*'
    end_pattern = r'(' + '|'.join(end_keywords) + r')'
    pattern = start_pattern + r'(.*?)' + end_pattern

    match = re.search(pattern, clean_text, re.IGNORECASE)
    if match:
        return match.group(2).strip()

    pattern = start_pattern + r'(.*)'
    match = re.search(pattern, clean_text, re.IGNORECASE)
    if match:
        return match.group(2).strip()

    return clean_text.strip()


# ---------------------------------------------------
# OCR NORMALIZATION
# ---------------------------------------------------
def normalize_ocr_text(text):
    text = text.lower()

    replacements = {
        "\n": " ",
        "|": " ",
        ":": " ",
        ";": " ",
        ",": ".",

        "nutrition facts": "nutritional information",
        "nutrition information": "nutritional information",
        "nutritional informationa": "nutritional information",

        "serve size": "serving size",
        "servesize": "serving size",
        "servingsize": "serving size",

        "per 100 g": "per 100g",
        "per100 g": "per 100g",
        "per100g": "per 100g",
        "100 g": "100g",

        "enery": "energy",
        "enerjy": "energy",
        "engry": "energy",
        "calorles": "calories",
        "calorie": "calories",
        "kcai": "kcal",
        "kca!": "kcal",
        "k cai": "kcal",
        "k j": "kj",

        "protien": "protein",
        "proten": "protein",

        "carbohydraate": "carbohydrate",
        "carbohydrale": "carbohydrate",
        "carbohydrat": "carbohydrate",
        "carbs": "carbohydrate",

        "suger": "sugar",
        "sugers": "sugars",
        "total sugar": "total sugars",

        "totai": "total",
        "tatol": "total",

        "saturaled": "saturated",
        "transfat": "trans fat",
        "trans fal": "trans fat",

        "fibre": "fiber",
        "dietaryfibre": "dietary fiber",
        "dietry fiber": "dietary fiber",
        "fibar": "fiber",

        "sodiurn": "sodium",
        "sodlum": "sodium",
        "sodum": "sodium",

        "gm": "g",
        "mg.": "mg",
        "oq": "0 g",
        "og": "0 g",
        "omg": "0 mg",
        "o mg": "0 mg",
        "o g": "0 g",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------
# COLUMN EXTRACTION: PER 100G ONLY
# ---------------------------------------------------
def extract_per_100g_text_from_words(words):
    if not words:
        return ""

    normalized_words = []
    for w in words:
        normalized_words.append({**w, "norm": normalize_ocr_text(w["text"])})

    per_candidates = []
    each_candidates = []

    for i, w in enumerate(normalized_words):
        nearby = " ".join(
            nw["norm"] for nw in normalized_words[max(0, i - 2): min(len(normalized_words), i + 4)]
        )
        if "per 100g" in nearby or "per100g" in nearby:
            per_candidates.append(w)

        if w["norm"] == "each":
            each_candidates.append(w)

    if not per_candidates:
        return ""

    per_word = min(per_candidates, key=lambda x: x["left"])
    per_x = per_word["left"]

    if each_candidates:
        each_word = min(each_candidates, key=lambda x: abs(x["top"] - per_word["top"]))
        each_x = each_word["left"]
        left_bound = max(0, per_x - 45)
        right_bound = each_x - 15
    else:
        left_bound = max(0, per_x - 45)
        right_bound = per_x + 240

    selected = []
    for w in normalized_words:
        center_x = (w["left"] + w["right"]) / 2
        if left_bound <= center_x <= right_bound:
            selected.append(w)

    if not selected:
        return ""

    selected.sort(key=lambda x: (x["top"], x["left"]))

    lines = {}
    for w in selected:
        key = (w["block_num"], w["par_num"], w["line_num"])
        lines.setdefault(key, []).append(w)

    ordered_lines = []
    for _, items in lines.items():
        items.sort(key=lambda x: x["left"])
        line_text = " ".join(item["norm"] for item in items)
        top = min(item["top"] for item in items)
        ordered_lines.append((top, line_text))

    ordered_lines.sort(key=lambda x: x[0])

    final_text = "\n".join(line for _, line in ordered_lines)
    return normalize_ocr_text(final_text)


# ---------------------------------------------------
# ROW-WISE TABLE PARSER
# ---------------------------------------------------
NUTRIENT_ALIASES = {
    "energy": ["energy", "calories", "calorie"],
    "protein": ["protein"],
    "carbohydrates": ["carbohydrate", "carbs"],
    "sugar": ["sugar", "sugars", "total sugar", "total sugars", "added sugar", "added sugars", "of which sugars"],
    "fat": ["fat", "total fat", "added fat"],
    "saturated_fat": ["saturated fat", "saturates", "of which saturates"],
    "trans_fat": ["trans fat"],
    "fiber": ["fiber", "fibre", "dietary fiber"],
    "sodium": ["sodium", "salt"],
}


def find_alias_position(text, aliases):
    best_pos = None
    best_alias = None

    for alias in aliases:
        match = re.search(rf'\b{re.escape(alias)}\b', text, re.IGNORECASE)
        if match:
            pos = match.start()
            if best_pos is None or pos < best_pos:
                best_pos = pos
                best_alias = alias

    return best_pos, best_alias


def extract_values_from_chunk(chunk):
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*(kcal|kj|mg|g)', chunk, re.IGNORECASE)
    return [(float(v), u.lower()) for v, u in matches]


def choose_best_value(field_name, values):
    if not values:
        return None

    if field_name == "energy":
        kcal_vals = [v for v, u in values if u == "kcal"]
        if kcal_vals:
            return kcal_vals[0]

        kj_vals = [v for v, u in values if u == "kj"]
        if kj_vals:
            return round(kj_vals[0] * 0.239, 2)

    if field_name == "sodium":
        mg_vals = [v for v, u in values if u == "mg"]
        if mg_vals:
            return mg_vals[0]

        g_vals = [v for v, u in values if u == "g"]
        if g_vals:
            return round(g_vals[0] * 1000, 2)

    g_vals = [v for v, u in values if u == "g"]
    if g_vals:
        return g_vals[0]

    return values[0][0]


def parse_from_per100g_text(per_100g_text):
    text = normalize_ocr_text(per_100g_text)
    if not text:
        return None

    lines = [line.strip() for line in per_100g_text.split("\n") if line.strip()]
    merged = []

    i = 0
    while i < len(lines):
        current = lines[i].strip()

        # merge split rows like "of which" + "sugars 1.9g"
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if (
                current in ["of", "of which", "which", "of which saturates", "of which sugars"]
                or (current.startswith("of which") and not re.search(r'\d', current))
            ):
                current = f"{current} {nxt}"
                i += 1

        merged.append(current)
        i += 1

    nutrition = {
        "energy": None,
        "protein": None,
        "carbohydrates": None,
        "sugar": None,
        "fat": None,
        "saturated_fat": None,
        "trans_fat": None,
        "fiber": None,
        "sodium": None,
    }

    for line in merged:
        line_norm = normalize_ocr_text(line)
        values = extract_values_from_chunk(line_norm)
        if not values:
            continue

        for field_name, aliases in NUTRIENT_ALIASES.items():
            pos, _ = find_alias_position(line_norm, aliases)
            if pos is not None:
                value = choose_best_value(field_name, values)
                if value is not None:
                    # prefer first meaningful capture
                    if nutrition[field_name] is None:
                        nutrition[field_name] = value
                break

    return nutrition


def parse_from_full_text(full_text):
    text = normalize_ocr_text(full_text)

    def get_window_value(field_name, aliases):
        for alias in aliases:
            match = re.search(rf'\b{re.escape(alias)}\b', text, re.IGNORECASE)
            if match:
                chunk = text[match.end(): match.end() + 90]
                values = extract_values_from_chunk(chunk)
                value = choose_best_value(field_name, values)
                if value is not None:
                    return value
        return None

    nutrition = {
        "energy": get_window_value("energy", NUTRIENT_ALIASES["energy"]),
        "protein": get_window_value("protein", NUTRIENT_ALIASES["protein"]),
        "carbohydrates": get_window_value("carbohydrates", NUTRIENT_ALIASES["carbohydrates"]),
        "sugar": get_window_value("sugar", NUTRIENT_ALIASES["sugar"]),
        "fat": get_window_value("fat", NUTRIENT_ALIASES["fat"]),
        "saturated_fat": get_window_value("saturated_fat", NUTRIENT_ALIASES["saturated_fat"]),
        "trans_fat": get_window_value("trans_fat", NUTRIENT_ALIASES["trans_fat"]),
        "fiber": get_window_value("fiber", NUTRIENT_ALIASES["fiber"]),
        "sodium": get_window_value("sodium", NUTRIENT_ALIASES["sodium"]),
    }

    return nutrition


def merge_nutrition(primary, fallback):
    final = {}
    for key in ["energy", "protein", "carbohydrates", "sugar", "fat", "saturated_fat", "trans_fat", "fiber", "sodium"]:
        final[key] = primary.get(key) if primary and primary.get(key) is not None else fallback.get(key)
    return final


# ---------------------------------------------------
# SCORE
# ---------------------------------------------------
def calculate_health_score(nutrition):
    score = 100
    reasons = []

    energy = nutrition.get("energy")
    sugar = nutrition.get("sugar")
    fat = nutrition.get("fat")
    saturated_fat = nutrition.get("saturated_fat")
    trans_fat = nutrition.get("trans_fat")
    sodium = nutrition.get("sodium")
    protein = nutrition.get("protein")
    fiber = nutrition.get("fiber")

    if energy is not None:
        if energy > 500:
            score -= 20
            reasons.append("High energy")
        elif energy > 300:
            score -= 10
            reasons.append("Moderate energy")

    if sugar is not None:
        if sugar > 20:
            score -= 18
            reasons.append("High sugar")
        elif sugar > 10:
            score -= 8
            reasons.append("Moderate sugar")

    if fat is not None:
        if fat > 30:
            score -= 15
            reasons.append("High fat")
        elif fat > 15:
            score -= 8
            reasons.append("Moderate fat")

    if saturated_fat is not None:
        if saturated_fat > 10:
            score -= 20
            reasons.append("High saturated fat")
        elif saturated_fat > 5:
            score -= 10
            reasons.append("Moderate saturated fat")

    if trans_fat is not None and trans_fat > 0:
        score -= 15
        reasons.append("Trans fat present")

    if sodium is not None:
        if sodium > 500:
            score -= 15
            reasons.append("High sodium or salt")
        elif sodium > 200:
            score -= 8
            reasons.append("Moderate sodium or salt")

    if protein is not None:
        if protein >= 8:
            score += 5
            reasons.append("Good protein")
        elif protein >= 4:
            score += 2
            reasons.append("Some protein")

    if fiber is not None:
        if fiber >= 5:
            score += 5
            reasons.append("Good fiber")
        elif fiber >= 2:
            score += 2
            reasons.append("Some fiber")

    score = max(0, min(100, round(score)))

    if score >= 80:
        label = "Good"
    elif score >= 60:
        label = "Moderate"
    elif score >= 40:
        label = "Risky"
    else:
        label = "Poor"

    found_fields = sum(1 for v in nutrition.values() if v is not None)

    return {
        "score": score,
        "label": label,
        "reasons": reasons if reasons else ["No strong nutrition indicators found"],
        "detected_fields": found_fields
    }

# ---------------------------------------------------
# PROFILE COMPARISON USING XLSX
# ---------------------------------------------------
def get_score_band(score):
    if score >= 80:
        return {"label": "Safe", "color": "#3ecf8e"}
    elif score >= 55:
        return {"label": "Moderate", "color": "#f5c842"}
    elif score >= 30:
        return {"label": "Caution", "color": "#f07d3e"}
    return {"label": "Unsafe", "color": "#e84b4b"}


def build_profile_comparison_from_nutrition(nutrition):
    """
    Uses your current OCR output and applies profile-specific logic.
    This gives profile-wise score comparison even when xlsx row match is not found.
    """
    energy = nutrition.get("energy") or 0
    sugar = nutrition.get("sugar") or 0
    fat = nutrition.get("fat") or 0
    saturated_fat = nutrition.get("saturated_fat") or 0
    trans_fat = nutrition.get("trans_fat") or 0
    sodium = nutrition.get("sodium") or 0
    protein = nutrition.get("protein") or 0
    fiber = nutrition.get("fiber") or 0
    carbs = nutrition.get("carbohydrates") or 0

    profile_scores = {}

    base_score_data = calculate_health_score(nutrition)
    base_score = base_score_data["score"]

    for profile_key in PROFILE_META.keys():
        score = float(base_score)
        reasons = []

        if profile_key == "diabetic":
            if sugar > 10:
                score -= 20
                reasons.append("Diabetic: high sugar")
            if carbs > 45:
                score -= 15
                reasons.append("Diabetic: high carbohydrates")

        elif profile_key == "heart_patient":
            if sodium > 400:
                score -= 25
                reasons.append("Heart: high sodium")
            if saturated_fat > 5:
                score -= 20
                reasons.append("Heart: high saturated fat")
            if trans_fat > 0.5:
                score -= 20
                reasons.append("Heart: trans fat present")

        elif profile_key == "athlete":
            if protein >= 25:
                score += 15
                reasons.append("Athlete: high protein bonus")
            if energy >= 300:
                score += 5
                reasons.append("Athlete: energy support")

        elif profile_key == "child":
            if sugar > 8:
                score -= 15
                reasons.append("Child: high sugar")
            if sodium > 300:
                score -= 10
                reasons.append("Child: high sodium")

        elif profile_key == "elderly":
            if sodium > 500:
                score -= 15
                reasons.append("Elderly: high sodium")
            if protein >= 12:
                score += 4
                reasons.append("Elderly: some protein support")

        elif profile_key == "weight_loss":
            if energy > 300:
                score -= 10
                reasons.append("Weight loss: high energy")
            if fat > 15:
                score -= 10
                reasons.append("Weight loss: high fat")
            if sugar > 12:
                score -= 10
                reasons.append("Weight loss: high sugar")
            if energy <= 150:
                score += 8
                reasons.append("Weight loss: low energy bonus")

        elif profile_key == "vegan":
            if fiber >= 4:
                score += 6
                reasons.append("Vegan: good fiber")
            if protein >= 10:
                score += 4
                reasons.append("Vegan: protein support")

        score = max(0, min(100, round(score, 1)))
        band = get_score_band(score)

        profile_scores[profile_key] = {
            "profile": profile_key,
            "label": PROFILE_META[profile_key]["label"],
            "score": score,
            "band": band["label"],
            "color": band["color"],
            "reasons": reasons
        }

    return profile_scores


def find_similar_foods_from_xlsx(nutrition, top_n=5):
    """
    Compare OCR nutrition with uploaded xlsx rows and return closest foods.
    Current xlsx supports score_general, score_diabetic, score_heart_patient,
    score_athlete, score_weight_loss.
    """
    df = load_health_score_dataset()

    if df.empty:
        return []

    required_cols = ["name", "calories_num", "fat_g", "protein_g", "fiber_g", "sugars_g", "sodium_mg"]
    for col in required_cols:
        if col not in df.columns:
            return []

    target_energy = nutrition.get("energy") if nutrition.get("energy") is not None else 0
    target_fat = nutrition.get("fat") if nutrition.get("fat") is not None else 0
    target_protein = nutrition.get("protein") if nutrition.get("protein") is not None else 0
    target_fiber = nutrition.get("fiber") if nutrition.get("fiber") is not None else 0
    target_sugar = nutrition.get("sugar") if nutrition.get("sugar") is not None else 0
    target_sodium = nutrition.get("sodium") if nutrition.get("sodium") is not None else 0

    work = df.copy()

    work["_distance"] = (
        (work["calories_num"].fillna(0) - target_energy).abs() * 0.20 +
        (work["fat_g"].fillna(0) - target_fat).abs() * 1.00 +
        (work["protein_g"].fillna(0) - target_protein).abs() * 1.20 +
        (work["fiber_g"].fillna(0) - target_fiber).abs() * 1.20 +
        (work["sugars_g"].fillna(0) - target_sugar).abs() * 1.50 +
        (work["sodium_mg"].fillna(0) - target_sodium).abs() * 0.03
    )

    top = work.sort_values("_distance").head(top_n)

    foods = []
    for _, row in top.iterrows():
        profile_scores = {}
        for profile_key, col_name in PROFILE_SCORE_COLUMN_MAP.items():
            if col_name in row and pd.notna(row[col_name]):
                score_val = float(row[col_name])
                band = get_score_band(score_val)
                profile_scores[profile_key] = {
                    "profile": profile_key,
                    "label": PROFILE_META[profile_key]["label"],
                    "score": round(score_val, 1),
                    "band": band["label"],
                    "color": band["color"]
                }

        foods.append({
            "name": row.get("name", "Unknown"),
            "distance": round(float(row["_distance"]), 2),
            "nutrition": {
                "energy": float(row.get("calories_num", 0) or 0),
                "fat": float(row.get("fat_g", 0) or 0),
                "protein": float(row.get("protein_g", 0) or 0),
                "fiber": float(row.get("fiber_g", 0) or 0),
                "sugar": float(row.get("sugars_g", 0) or 0),
                "sodium": float(row.get("sodium_mg", 0) or 0),
            },
            "profile_scores": profile_scores
        })

    return foods

# ---------------------------------------------------
# NEW ADDITIONS ONLY
# ---------------------------------------------------
def preprocess_ingredients_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not loaded properly.")

    h, w = img.shape[:2]
    img = img[0:int(h * 0.92), :]

    h, w = img.shape[:2]
    img = img[int(h * 0.02):int(h * 0.88), int(w * 0.03):int(w * 0.98)]

    img = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12
    )

    return thresh, gray


def extract_ingredients_text_from_image(image_path):
    thresh, gray = preprocess_ingredients_image(image_path)

    configs = [
        r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
        r'--oem 3 --psm 4 -c preserve_interword_spaces=1',
        r'--oem 3 --psm 11 -c preserve_interword_spaces=1'
    ]

    best_text = ""
    best_conf = -1

    for img in [thresh, gray]:
        for config in configs:
            data = pytesseract.image_to_data(
                img,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            words = []
            confs = []

            for i in range(len(data["text"])):
                word = data["text"][i].strip()
                conf = data["conf"][i]

                try:
                    conf = float(conf)
                except Exception:
                    conf = -1

                if word and conf > 0:
                    words.append(word)
                    confs.append(conf)

            text = " ".join(words)
            avg_conf = sum(confs) / len(confs) if confs else 0

            if avg_conf > best_conf and len(text) > 20:
                best_conf = avg_conf
                best_text = text

    if not best_text:
        best_text = pytesseract.image_to_string(
            thresh,
            config='--oem 3 --psm 6'
        )

    return best_text.strip()


def calculate_ingredients_scan_score(full_text, ingredients_text):
    score = 0
    reasons = []

    full_len = len(full_text.strip()) if full_text else 0
    ing_len = len(ingredients_text.strip()) if ingredients_text else 0

    if full_len > 30:
        score += 25
        reasons.append("OCR text extracted")

    if ing_len > 20:
        score += 35
        reasons.append("Ingredients section found")

    if ing_len > 60:
        score += 20
        reasons.append("Ingredients text length is good")

    if re.search(r'\b(ingredients|ingredient|ingredlents|ingrediants|ingrdients)\b', full_text or "", re.IGNORECASE):
        score += 20
        reasons.append("Ingredients keyword detected")

    score = max(0, min(100, score))

    if score >= 80:
        label = "Strong"
    elif score >= 60:
        label = "Good"
    elif score >= 40:
        label = "Moderate"
    else:
        label = "Weak"

    return {
        "score": score,
        "label": label,
        "reasons": reasons if reasons else ["No strong ingredients section found"]
    }


def save_uploaded_file(file_obj):
    if not file_obj or file_obj.filename == "":
        raise ValueError("No selected file")
    filename = secure_filename(file_obj.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file_obj.save(file_path)
    return file_path


def run_nutrition_pipeline(file_path):
    full_text, color_img, gray_img, thresh_img = extract_text_from_image(file_path)
    ingredients_text = extract_ingredients_section(full_text)

    words = extract_word_data(thresh_img)
    per_100g_text = extract_per_100g_text_from_words(words)

    per100g_nutrition = parse_from_per100g_text(per_100g_text) if per_100g_text else None
    fulltext_nutrition = parse_from_full_text(full_text)

    nutrition = merge_nutrition(per100g_nutrition or {}, fulltext_nutrition or {})
    score_data = calculate_health_score(nutrition)

    profile_scores = build_profile_comparison_from_nutrition(nutrition)
    similar_foods = find_similar_foods_from_xlsx(nutrition, top_n=5)

    return {
        "message": "Nutrition image analyzed successfully",
        "mode": "nutrition",
        "full_text": full_text,
        "per_100g_text": per_100g_text,
        "ingredients_text": ingredients_text,
        "nutrition_data": nutrition,
        "score_prediction": score_data,
        "profile_scores": profile_scores,
        "similar_foods": similar_foods
    }


def run_ingredients_pipeline(file_path):
    full_text = extract_ingredients_text_from_image(file_path)
    ingredients_text = extract_ingredients_section(full_text)
    score_data = calculate_ingredients_scan_score(full_text, ingredients_text)

    return {
        "message": "Ingredients image analyzed successfully",
        "mode": "ingredients",
        "full_text": full_text,
        "ingredients_text": ingredients_text,
        "ingredients_score": score_data
    }


# ---------------------------------------------------
# API
# ---------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Backend is running successfully",
        "available_routes": [
            "/analyze [POST]",
            "/analyze-nutrition [POST]",
            "/analyze-ingredients [POST]",
            "/analyze-dual [POST]",
            "/profiles [GET]"
        ]
    })

@app.route("/analyze", methods=["GET", "POST"])
def analyze_image():
    if request.method == "GET":
        return jsonify({
            "message": "Use POST with form-data",
            "required_field": "image"
        })

    print("FILES:", request.files)
    print("FILE KEYS:", list(request.files.keys()))

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        full_text, color_img, gray_img, thresh_img = extract_text_from_image(file_path)
        ingredients_text = extract_ingredients_section(full_text)

        words = extract_word_data(thresh_img)
        per_100g_text = extract_per_100g_text_from_words(words)

        per100g_nutrition = parse_from_per100g_text(per_100g_text) if per_100g_text else None
        fulltext_nutrition = parse_from_full_text(full_text)

        nutrition = merge_nutrition(per100g_nutrition or {}, fulltext_nutrition or {})
        score_data = calculate_health_score(nutrition)

        return jsonify({
            "message": "Image analyzed successfully",
            "full_text": full_text,
            "per_100g_text": per_100g_text,
            "ingredients_text": ingredients_text,
            "nutrition_data": nutrition,
            "score_prediction": score_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-nutrition", methods=["GET", "POST"])
def analyze_nutrition_only():
    if request.method == "GET":
        return jsonify({
            "message": "Use POST with form-data",
            "required_field": "image"
        })

    print("FILES:", request.files)
    print("FILE KEYS:", list(request.files.keys()))

    if "image" not in request.files:
        return jsonify({"error": "No nutrition image file provided"}), 400

    try:
        file_path = save_uploaded_file(request.files["image"])
        result = run_nutrition_pipeline(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-ingredients", methods=["GET", "POST"])
def analyze_ingredients_only():
    print("REQUEST METHOD:", request.method)
    print("FILES:", request.files)
    print("FILE KEYS:", list(request.files.keys()))

    if request.method == "GET":
        return jsonify({
            "message": "Use POST with form-data",
            "required_field": "image"
        })

    if "image" not in request.files:
        return jsonify({
            "error": "No ingredients image file provided",
            "received_keys": list(request.files.keys())
        }), 400

    try:
        file_path = save_uploaded_file(request.files["image"])
        result = run_ingredients_pipeline(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-dual", methods=["POST"])
def analyze_dual():
    if "nutrition_image" not in request.files and "ingredients_image" not in request.files:
        return jsonify({"error": "Please upload at least one image"}), 400

    try:
        response_data = {
            "message": "Dual scan completed successfully"
        }

        if "nutrition_image" in request.files and request.files["nutrition_image"].filename != "":
            nutrition_path = save_uploaded_file(request.files["nutrition_image"])
            response_data["nutrition_result"] = run_nutrition_pipeline(nutrition_path)

        if "ingredients_image" in request.files and request.files["ingredients_image"].filename != "":
            ingredients_path = save_uploaded_file(request.files["ingredients_image"])
            response_data["ingredients_result"] = run_ingredients_pipeline(ingredients_path)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/profiles", methods=["GET"])
def get_profiles():
    return jsonify({
        "profiles": [
            {"key": "general", "label": "🧑 General"},
            {"key": "diabetic", "label": "🩺 Diabetic"},
            {"key": "heart_patient", "label": "❤️ Heart Patient"},
            {"key": "athlete", "label": "🏋️ Athlete"},
            {"key": "child", "label": "👶 Child"},
            {"key": "elderly", "label": "👴 Elderly"},
            {"key": "weight_loss", "label": "⚖️ Weight Loss"},
            {"key": "vegan", "label": "🌱 Vegan"},
        ]
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)