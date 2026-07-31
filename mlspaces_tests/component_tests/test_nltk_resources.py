import pytest

from molmo_spaces.utils import nltk_resources


def test_require_nltk_resources_accepts_local_directory_or_zip(monkeypatch):
    calls = []

    def fake_find(path):
        calls.append(path)
        if path in {"corpora/wordnet.zip", "corpora/wordnet2022"}:
            return path
        raise LookupError(path)

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)

    nltk_resources.require_nltk_resources()

    assert calls == [
        "corpora/wordnet",
        "corpora/wordnet.zip",
        "corpora/wordnet2022",
    ]


def test_require_nltk_resources_fails_without_downloading(monkeypatch):
    def fake_find(path):
        raise LookupError(path)

    def fail_download(*args, **kwargs):
        pytest.fail("runtime resource validation must not download")

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fail_download)

    with pytest.raises(
        LookupError,
        match=r"Missing required NLTK resources: wordnet, wordnet2022.*nltk\.downloader",
    ):
        nltk_resources.require_nltk_resources()
