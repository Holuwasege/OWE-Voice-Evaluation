import numpy as np

# ==========================
# TONE MAP FOR YORÙBÁ
# ==========================
TONE_MAP = {
    'á': ('a', 'H'), 'à': ('a', 'L'), 'a': ('a', 'M'),
    'é': ('e', 'H'), 'è': ('e', 'L'), 'e': ('e', 'M'),
    'ẹ́': ('ẹ', 'H'), 'ẹ̀': ('ẹ', 'L'), 'ẹ': ('ẹ', 'M'),
    'í': ('i', 'H'), 'ì': ('i', 'L'), 'i': ('i', 'M'),
    'ó': ('o', 'H'), 'ò': ('o', 'L'), 'o': ('o', 'M'),
    'ọ́': ('ọ', 'H'), 'ọ̀': ('ọ', 'L'), 'ọ': ('ọ', 'M'),
    'ú': ('u', 'H'), 'ù': ('u', 'L'), 'u': ('u', 'M')
}

# ====================================
# Extract tone sequence from a string
# ====================================
def extract_tone_sequence(text):
    tones = []
    i = 0
    while i < len(text):
        # handle 'ẹ́', 'ẹ̀', 'ọ́', 'ọ̀'
        if i+1 < len(text) and text[i:i+2] in TONE_MAP:
            _, tone = TONE_MAP[text[i:i+2]]
            tones.append(tone)
            i += 2
            continue

        # single-character tones
        if text[i] in TONE_MAP:
            _, tone = TONE_MAP[text[i]]
            tones.append(tone)

        i += 1

    return tones

# ==========================
# Levenshtein distance
# ==========================
def compute_levenshtein(ref, hyp):
    dp = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)

    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[-1][-1]

# ==========================
# Tone Error Rate (TER)
# ==========================
def tone_error_rate(references, predictions):
    total_tones = 0
    total_errors = 0

    for ref, pred in zip(references, predictions):
        ref_tones = extract_tone_sequence(ref)
        pred_tones = extract_tone_sequence(pred)

        total_tones += len(ref_tones)
        total_errors += compute_levenshtein(ref_tones, pred_tones)

    return total_errors / total_tones if total_tones > 0 else 0.0




