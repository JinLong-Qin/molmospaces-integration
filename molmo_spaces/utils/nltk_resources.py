import nltk


NLTK_RESOURCE_PATHS = {
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
    "wordnet2022": ("corpora/wordnet2022", "corpora/wordnet2022.zip"),
}


def require_nltk_resources() -> None:
    """Require locally prepared corpora without performing network I/O."""
    missing = []
    for corpus, candidates in NLTK_RESOURCE_PATHS.items():
        for candidate in candidates:
            try:
                nltk.data.find(candidate)
                break
            except LookupError:
                continue
        else:
            missing.append(corpus)

    if missing:
        names = ", ".join(missing)
        raise LookupError(
            f"Missing required NLTK resources: {names}. "
            "Download them during setup with "
            "`python -m nltk.downloader wordnet wordnet2022`."
        )
