from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import importlib

# =========================
# CONFIG
# =========================
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"
OUTLIER_THRESHOLD = 0.5

# =========================
# MODELE
# =========================
model = SentenceTransformer(MODEL_NAME)


# =========================
# UTILS
# =========================
def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


# =========================
# CENTROIDS
# =========================
def compute_centroids(threshold: float = OUTLIER_THRESHOLD):
    """
    Retourne :
        - intent_centroids : dict[intent -> np.array]
        - sub_centroids    : dict[intent -> dict[sub_intent -> np.array]]
        - imports          : dict[intent -> dict[sub_intent -> module]]
    """
    base = Path("./modules")

    intent_centroids = {}
    sub_centroids = {}
    imports = {}

    for dossier in base.iterdir():
        if not dossier.is_dir() or dossier.name == "__pycache__":
            continue
        if not (dossier / "__init__.py").is_file():
            continue

        data_path = dossier / "data.json"
        if not data_path.is_file():
            continue

        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)

        sub_phrases = {}
        sub_imports = {}

        for sub_name, phrases in data.items():
            py_file = dossier / f"{sub_name}.py"
            if py_file.is_file():
                sub_phrases[sub_name] = phrases
                sub_imports[sub_name] = importlib.import_module(
                    f"modules.{dossier.name}.{sub_name}"
                )

        if not sub_phrases:
            continue

        # =========================
        # INTENT CENTROID (ROBUST)
        # =========================
        all_phrases = [
            phrase
            for phrases in sub_phrases.values()
            for phrase in phrases
        ]

        embeddings = model.encode(all_phrases, normalize_embeddings=True)

        temp_centroid = np.mean(embeddings, axis=0)
        sims = util.cos_sim(embeddings, temp_centroid).flatten()

        mask = sims >= threshold
        filtered = embeddings[mask] if mask.any() else embeddings

        intent_centroids[dossier.name] = normalize(
            np.mean(filtered, axis=0)
        )

        # =========================
        # SUB-INTENT CENTROIDS
        # =========================
        sub_centroids[dossier.name] = {}

        for sub_name, phrases in sub_phrases.items():
            sub_emb = model.encode(phrases, normalize_embeddings=True)

            temp_sub = np.mean(sub_emb, axis=0)
            sub_sims = util.cos_sim(sub_emb, temp_sub).flatten()

            sub_mask = sub_sims >= threshold
            filtered_sub = sub_emb[sub_mask] if sub_mask.any() else sub_emb

            sub_centroids[dossier.name][sub_name] = normalize(
                np.mean(filtered_sub, axis=0)
            )

        imports[dossier.name] = sub_imports

    return intent_centroids, sub_centroids, imports
