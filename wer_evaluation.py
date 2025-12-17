
def asr_error_analysis(
    references,
    hypotheses,
    normalize=True
):
    """
    Perform word-level ASR error analysis.

    Args:
        references (list[str]): Ground-truth sentences
        hypotheses (list[str]): ASR model predictions
        normalize (bool): Whether to apply Yoruba normalization

    Returns:
        dict containing:
            - wer
            - substitutions
            - deletions
            - insertions
    """

    # --- Normalization function ---
    def normalize_yoruba(text):
        text = str(text).lower()
        text = text.replace("’", "'")
        text = text.strip()
        return text

    if normalize:
        references = [normalize_yoruba(s) for s in references]
        hypotheses = [normalize_yoruba(s) for s in hypotheses]

    # --- Run alignment ---
    measures = process_words(references, hypotheses)

    substitutions = []
    deletions = []
    insertions = []

    # --- Iterate sentence by sentence ---
    for i in range(len(references)):
        ref_words = measures.references[i]
        hyp_words = measures.hypotheses[i]
        alignment_ops = measures.alignments[i]

        for op in alignment_ops:
            if op.type == "substitute":
                ref_seg = ref_words[op.ref_start_idx:op.ref_end_idx]
                hyp_seg = hyp_words[op.hyp_start_idx:op.hyp_end_idx]
                substitutions.append(
                    (" ".join(ref_seg), " ".join(hyp_seg))
                )

            elif op.type == "delete":
                ref_seg = ref_words[op.ref_start_idx:op.ref_end_idx]
                deletions.append(" ".join(ref_seg))

            elif op.type == "insert":
                hyp_seg = hyp_words[op.hyp_start_idx:op.hyp_end_idx]
                insertions.append(" ".join(hyp_seg))

    return {
        "wer": measures.wer,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions
    }
