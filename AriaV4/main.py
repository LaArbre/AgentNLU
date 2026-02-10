from AriaV4.nlu import detect_intents
import json

# =========================
# LOAD TEST QUERIES
# =========================
with open("../AriaV3/test_queries.json", "r", encoding="utf-8") as f:
    TEST_QUERIES = json.load(f)

# =========================
# TEST AUTOMATIQUE
# =========================
if __name__ == "__main__":
    count_true = 0
    count_none = 0
    count_error = 0

    for entry in TEST_QUERIES:
        phrase = entry["phrase"]
        expected = entry["intent"]

        result = detect_intents(phrase, debug=False)
        detected_intents = list(result.keys())

        # ----- intent attendu = None -----
        if expected is None:
            if not detected_intents:
                count_true += 1
            else:
                count_error += 1
                print("=" * 50)
                detect_intents(phrase, debug=True)

        # ----- intent attendu défini -----
        else:
            if detected_intents == [expected]:
                count_true += 1
            else:
                if detected_intents:
                    count_error += 1
                else:
                    count_none += 1

                print("=" * 50)
                detect_intents(phrase, debug=True)

    # =========================
    # STATS
    # =========================
    total = len(TEST_QUERIES)
    print("=" * 50)
    print(f"Taux réussite : {count_true / total * 100:05.2f} % | {count_true}")
    print(f"Taux erreur   : {count_error / total * 100:05.2f} % | {count_error}")
    print(f"Taux None     : {count_none / total * 100:05.2f} % | {count_none}")
    print("=" * 50)

    # =========================
    # MODE INTERACTIF
    # =========================
    while True:
        phrase = input(">>> ")
        if phrase.lower() in {"exit", "quit", "q"}:
            break

        detect_intents(phrase, debug=True)
        print("=" * 50)
