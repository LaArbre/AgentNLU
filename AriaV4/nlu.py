from sentence_transformers import SentenceTransformer, util
from .text_processing import split_clauses
from modules.main import compute_centroids
import numpy as np

# =========================
# CONFIG
# =========================
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"

TEMPERATURE = 0.7

MIN_PROB = 0.45
MIN_MARGIN = 0.03
MIN_ENERGY = 0.02

INTENT_SIM_THRESHOLD = 0.75
INTENT_SIM_PENALTY = 0.15

# =========================
# MODELE
# =========================
model = SentenceTransformer(MODEL_NAME)

# =========================
# INIT
# =========================
INTENT_CENTROIDS, SUB_CENTROIDS, IMPORTS = compute_centroids()


# =========================
# OUTILS
# =========================
def softmax(scores: list[float], temperature: float) -> list[float]:
    x = np.array(scores) / temperature
    x -= np.max(x)
    exp_x = np.exp(x)
    return (exp_x / exp_x.sum()).tolist()


def compute_intent_similarities(centroids: dict):
    sims = {}
    intents = list(centroids.keys())

    for i, a in enumerate(intents):
        for b in intents[i + 1:]:
            sim = util.cos_sim(centroids[a], centroids[b]).item()
            sims[(a, b)] = sim
            sims[(b, a)] = sim

    return sims


INTENT_SIMILARITIES = compute_intent_similarities(INTENT_CENTROIDS)


def apply_penalty(scores, intents):
    adjusted = scores.copy()

    for i, a in enumerate(intents):
        for j, b in enumerate(intents):
            if i == j:
                continue
            sim = INTENT_SIMILARITIES.get((a, b), 0.0)
            if sim >= INTENT_SIM_THRESHOLD:
                adjusted[i] -= sim * INTENT_SIM_PENALTY

    return adjusted


# =========================
# PIPELINE NLU
# =========================
def detect_intents(query: str, debug: bool = True):
    final_intents = {}
    clauses = split_clauses(query)

    if debug:
        print(f"[DEBUG] Phrase : {query}")
        print(f"[DEBUG] Clauses : {clauses}")

    for idx, clause in enumerate(clauses, 1):
        emb = model.encode(clause, normalize_embeddings=True)

        intent_names = []
        raw_scores = []

        for intent, centroid in INTENT_CENTROIDS.items():
            intent_names.append(intent)
            raw_scores.append(util.cos_sim(emb, centroid).item())

        penalized_scores = apply_penalty(raw_scores, intent_names)
        probs = softmax(penalized_scores, TEMPERATURE)

        best_idx = int(np.argmax(probs))
        best_intent = intent_names[best_idx]
        best_prob = probs[best_idx]

        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]
        mean_prob = np.mean(probs)
        energy = best_prob - mean_prob

        if debug:
            print(
                f"[CLAUSE_{idx}] intent={best_intent} | "
                f"prob={best_prob:.3f} | "
                f"margin={margin:.3f} | "
                f"energy={energy:.3f} | ",
                end=""
            )

        if best_prob >= MIN_PROB and margin >= MIN_MARGIN and energy >= MIN_ENERGY:
            best_sub = None
            best_sub_score = -1

            for sub, centroid in SUB_CENTROIDS[best_intent].items():
                s = util.cos_sim(emb, centroid).item()
                if s > best_sub_score:
                    best_sub_score = s
                    best_sub = sub

            callable_module = IMPORTS[best_intent][best_sub]

            final_intents[best_intent] = {
                "probability": best_prob,
                "submodule": best_sub,
                "sub_score": best_sub_score,
                "callable": callable_module
            }

            if debug:
                print(f"ACCEPT | sub={best_sub} ({best_sub_score:.3f})")

            if hasattr(callable_module, "main"):
                if debug:
                    print(f"[CALL] {best_intent}.{best_sub}.main()")
                callable_module.main()

        else:
            if debug:
                print("REJECT")

    return final_intents
