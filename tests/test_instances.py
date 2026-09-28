"""Offline tests for benchmark-instance construction and the review CSV."""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))

from test_data import (  # noqa: E402
    CAND_COLS,
    SENT_COLS,
    TEXT_COLS,
    WORD_COLS,
    build_fixture,
    write_zst_csv,
)
from core.data import read_corpus_tables  # noqa: E402
from core.instances import (  # noqa: E402
    Candidate,
    CorpusTagError,
    Instance,
    build_language_instances,
    determine_data_tag,
    has_empty_candidate_gramset,
    review_csv_path,
    write_review_csv,
)

TEXTS = [(1, "", "", "", "")]
SENTENCES = [
    (10, 1, '<s id="10"><w id="1">one</w></s>', ""),
    (11, 1, '<s id="11"><w id="2">two</w></s>', ""),
    (12, 1, '<s id="12"><w id="3">three</w></s>', ""),
]


def write_corpus(root: Path, lang: str, texts, sentences, words, candidates) -> Path:
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    write_zst_csv(corpus / f"texts_{lang}.csv.zst", TEXT_COLS, texts)
    write_zst_csv(corpus / f"sentences_{lang}.csv.zst", SENT_COLS, sentences)
    write_zst_csv(corpus / f"words_{lang}.csv.zst", WORD_COLS, words)
    write_zst_csv(corpus / f"candidate_analyses_{lang}.csv.zst", CAND_COLS, candidates)
    return root


def build(root: Path, words, candidates) -> str:
    write_corpus(root, "krl", TEXTS, SENTENCES, words, candidates)
    return read_corpus_tables("krl", root)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SRC / "cli.py"), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_same_wordform_different_gramset_is_two_analyses(tmp_path):
    words = [(100, 10, 1, "tere")]
    candidates = [(100, 9001, "SG+NOM", 2), (100, 9001, "SG+ACC", 1)]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.funnel[6].retained == 1
    instance = result.instances[0]
    assert instance.candidates == (
        Candidate(9001, "SG+ACC"),
        Candidate(9001, "SG+NOM"),
    )
    assert instance.gold_analysis == Candidate(9001, "SG+NOM")
    assert instance.word == "tere"
    assert instance.sentence_id == 10
    assert instance.text_id == 1
    assert instance.sentence_xml == '<s id="10"><w id="1">one</w></s>'


def test_identical_candidate_rows_are_one_analysis(tmp_path):
    words = [(100, 10, 1, "tere")]
    candidates = [(100, 9001, "SG+NOM", 2), (100, 9001, "SG+NOM", 1)]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.funnel[1].retained == 1
    assert result.funnel[2].retained == 0
    assert result.instances == ()


def test_duplicate_identity_excludes_otherwise_eligible_word(tmp_path):
    words = [(100, 10, 1, "tere")]
    candidates = [
        (100, 9001, "SG+NOM", 2),
        (100, 9001, "SG+NOM", 1),
        (100, 9002, "SG+GEN", 1),
    ]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.funnel[1].retained == 1
    assert result.funnel[2].retained == 1
    assert result.funnel[3].retained == 0
    assert result.step1_words_with_duplicate_identity == 1
    assert result.instances == ()


def test_relevance_one_is_not_gold(tmp_path):
    words = [(200, 10, 1, "zero-gold"), (201, 11, 1, "two-gold")]
    candidates = [
        (200, 9003, "INTERJ", 0),
        (200, 9004, "N+SG+NOM", 1),
        (201, 9005, "A", 1),
        (201, 9006, "B", 2),
        (201, 9007, "C", 2),
    ]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.funnel[0].retained == 2
    assert result.funnel[1].retained == 0
    assert result.instances == ()


def test_selected_candidate_with_empty_gramset_excludes_word(tmp_path):
    words = [(100, 10, 1, "tere")]
    candidates = [(100, 9001, "", 2), (100, 9002, "SG+GEN", 1)]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.funnel[4].retained == 0
    assert result.step1_words_with_selected_empty_gramset == 1
    assert result.instances == ()


def test_unselected_empty_gramset_removed_at_step7(tmp_path):
    words = [(100, 10, 1, "tere")]
    candidates = [(100, 9001, "", 1), (100, 9002, "SG+ACC", 2)]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.funnel[6].retained == 1
    assert result.funnel[7].retained == 0
    assert result.funnel[7].removed == 1
    assert result.instances == ()
    assert result.step1_words_with_unselected_empty_gramset == 1


def test_selected_empty_gramset_not_recounted_at_step7(tmp_path):
    words = [(100, 10, 1, "tere"), (101, 11, 1, "ok")]
    candidates = [
        (100, 9001, "", 2), (100, 9002, "SG+GEN", 1),
        (101, 9003, "A", 1), (101, 9004, "B", 2),
    ]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.funnel[4].retained == 1
    assert result.funnel[6].retained == 1
    assert result.funnel[7].retained == 1
    assert result.funnel[7].removed == 0
    assert [inst.word_id for inst in result.instances] == [101]


def test_clean_instance_keeps_full_candidate_tuple(tmp_path):
    words = [(100, 10, 1, "tere")]
    candidates = [
        (100, 9001, "SG+ACC", 1),
        (100, 9002, "SG+NOM", 2),
        (100, 9003, "SG+GEN", 1),
    ]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.funnel[6].retained == 1
    assert result.funnel[7].retained == 1
    assert len(result.instances) == result.funnel[7].retained
    assert result.instances[0].candidates == (
        Candidate(9001, "SG+ACC"),
        Candidate(9002, "SG+NOM"),
        Candidate(9003, "SG+GEN"),
    )
    assert result.instances[0].gold_analysis == Candidate(9002, "SG+NOM")


def test_step7_removed_equals_step6_minus_step7(tmp_path):
    words = [(100, 10, 1, "a"), (101, 11, 1, "b"), (102, 12, 1, "c")]
    candidates = [
        (100, 9001, "", 1), (100, 9002, "SG+ACC", 2),
        (101, 9003, "A", 1), (101, 9004, "B", 2),
        (102, 9005, "C", 1), (102, 9006, "D", 2),
    ]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.funnel[7].removed == result.funnel[6].retained - result.funnel[7].retained
    assert len(result.instances) == result.funnel[7].retained
    assert [inst.word_id for inst in result.instances] == [101, 102]


def test_has_empty_candidate_gramset():
    assert not has_empty_candidate_gramset(
        (Candidate(1, "A"), Candidate(2, "B"))
    )
    assert has_empty_candidate_gramset(
        (Candidate(1, "A"), Candidate(2, ""))
    )


def test_zero_position_word_excludes_whole_sentence(tmp_path):
    words = [(100, 10, 0, "zero"), (101, 10, 1, "target"), (102, 11, 1, "ok")]
    candidates = [
        (101, 9001, "SG+NOM", 2),
        (101, 9002, "SG+GEN", 1),
        (102, 9003, "A", 1),
        (102, 9004, "B", 2),
    ]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.zero_position_sentences == frozenset({10})
    assert result.step1_words_with_zero_position_sentence == 1
    assert result.funnel[6].retained == 1
    assert [inst.word_id for inst in result.instances] == [102]
    assert all(inst.word_number > 0 for inst in result.instances)


def test_repeated_positive_position_excludes_whole_sentence(tmp_path):
    words = [
        (101, 10, 1, "a"),
        (102, 10, 1, "b"),
        (103, 11, 1, "c"),
        (104, 11, 2, "d"),
    ]
    candidates = [
        (101, 9001, "SG+NOM", 2),
        (101, 9002, "SG+GEN", 1),
        (102, 9003, "A", 1),
        (102, 9004, "B", 2),
        (103, 9005, "C", 1),
        (103, 9006, "D", 2),
        (104, 9007, "E", 1),
        (104, 9008, "F", 2),
    ]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.repeated_position_sentences == frozenset({10})
    assert result.step1_words_with_repeated_position_sentence == 2
    assert result.funnel[6].retained == 2
    assert all(inst.sentence_id == 11 for inst in result.instances)


def test_unavailable_context_never_creates_instance(tmp_path):
    words = [
        (100, 999, 1, "no-sentence"),
        (101, 10, 1, "dup-sentence"),
        (102, 11, 1, "no-text"),
        (103, 12, 1, "dup-text"),
    ]
    candidates = [
        (100, 9001, "SG+NOM", 2), (100, 9002, "SG+GEN", 1),
        (101, 9003, "A", 1), (101, 9004, "B", 2),
        (102, 9005, "C", 1), (102, 9006, "D", 2),
        (103, 9007, "E", 1), (103, 9008, "F", 2),
    ]
    texts = [(1, "", "", "", ""), (2, "", "", "", ""), (2, "", "", "", "")]
    sentences = [
        (10, 1, "<s>one</s>", ""),
        (10, 1, "<s>one dup</s>", ""),
        (11, 777, "<s>two</s>", ""),
        (12, 2, "<s>three</s>", ""),
    ]
    write_corpus(tmp_path, "krl", texts, sentences, words, candidates)
    result = build_language_instances("krl", read_corpus_tables("krl", tmp_path))
    assert result.funnel[5].retained == 0
    assert result.instances == ()
    assert result.step1_words_with_unavailable_sentence == 4


def test_review_csv_is_exact_schema_sorted_and_unique(tmp_path):
    path = tmp_path / "out" / "zero_word_number_sentences_v2026.09.csv"
    rows = [("lud", 3), ("krl", 10), ("krl", 2), ("vep", 5), ("krl", 2)]
    written = write_review_csv(path, rows)
    assert written == 4
    with open(path, encoding="utf-8", newline="") as fh:
        lines = list(csv.reader(fh))
    assert lines[0] == ["language", "sentence_id"]
    assert lines[1:] == [["krl", "2"], ["krl", "10"], ["lud", "3"], ["vep", "5"]]


def test_review_rows_include_words_without_candidates_or_sentence(tmp_path):
    words = [(100, 10, 0, "zero-plain"), (101, 999, 0, "zero-absent-sentence")]
    candidates = []
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.zero_position_sentences == frozenset({10, 999})
    assert result.funnel[0].retained == 2
    assert result.funnel[1].retained == 0
    assert review_csv_path(tmp_path, "v1").name == "zero_word_number_sentences_v1.csv"


def test_funnel_monotonic_and_removed_consistent(tmp_path):
    words = [
        (100, 10, 1, "a"),
        (101, 11, 1, "b"),
        (102, 12, 1, "c"),
        (103, 13, 1, "d"),
        (104, 14, 1, "e"),
        (105, 15, 1, "f"),
        (106, 16, 1, "g"),
        (107, 20, 1, "h"),
        (108, 20, 0, "zero-sibling"),
    ]
    candidates = [
        (100, 9001, "X1", 2), (100, 9002, "X2", 1),
        (101, 9011, "Y1", 0),
        (102, 9021, "Z1", 2), (102, 9022, "Z2", 2),
        (103, 9031, "P1", 2), (103, 9031, "P1", 1), (103, 9032, "P2", 1),
        (104, 9041, "", 2), (104, 9042, "Q2", 1),
        (105, 9051, "R1", 2), (105, 9052, "R2", 1),
        (106, 9061, "S1", 2), (106, 9062, "S2", 1),
        (107, 9071, "T1", 2), (107, 9072, "T2", 1),
    ]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    retained = [step.retained for step in result.funnel]
    assert [step.index for step in result.funnel] == list(range(8))
    assert retained == sorted(retained, reverse=True)
    for index in range(1, len(retained)):
        assert result.funnel[index].removed == retained[index - 1] - retained[index]
    assert retained[7] == 1
    assert result.instances[0].word_id == 100


def test_word_with_two_anomalies_not_double_subtracted(tmp_path):
    words = [
        (300, 30, 0, "z"),
        (301, 30, 1, "r1"),
        (302, 30, 1, "r2"),
        (303, 31, 1, "clean"),
    ]
    candidates = [
        (301, 9001, "SG+NOM", 2), (301, 9002, "SG+GEN", 1),
        (302, 9003, "A", 1), (302, 9004, "B", 2),
        (303, 9005, "C", 1), (303, 9006, "D", 2),
    ]
    write_corpus(
        tmp_path,
        "krl",
        TEXTS,
        [
            (30, 1, '<s id="30">murre</s>', ""),
            (31, 1, '<s id="31">selkeä</s>', ""),
        ],
        words,
        candidates,
    )
    result = build_language_instances("krl", read_corpus_tables("krl", tmp_path))
    assert result.step1_words_with_zero_position_sentence == 2
    assert result.step1_words_with_repeated_position_sentence == 2
    assert result.funnel[5].retained == 3
    assert result.funnel[6].retained == 1
    assert result.funnel[6].removed == 2
    assert [inst.word_id for inst in result.instances] == [303]


def test_instances_are_materialized_for_surviving_words(tmp_path):
    words = [
        (100, 10, 1, "keep"),
        (101, 11, 1, "keep2"),
    ]
    candidates = [
        (100, 9001, "A1", 2), (100, 9002, "A2", 1), (100, 9003, "A3", 1),
        (101, 9011, "B1", 2), (101, 9012, "B2", 1),
    ]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert len(result.instances) == 2
    assert all(isinstance(inst, Instance) for inst in result.instances)
    assert [inst.language for inst in result.instances] == ["krl", "krl"]


def test_grouped_materialization_unsorted_source_rows(tmp_path):
    words = [
        (100, 10, 1, "tere"),
        (102, 12, 1, "kolmas"),
        (101, 11, 1, "teine"),
    ]
    candidates = [
        (102, 9011, "SG+ACC", 1),
        (102, 9011, "SG+NOM", 2),
        (102, 9012, "SG+GEN", 1),
        (100, 9001, "SG+NOM", 2),
        (100, 9001, "SG+ACC", 1),
        (100, 9002, "SG+GEN", 1),
        (101, 9005, "A", 1),
        (101, 9006, "B", 1),
    ]
    result = build_language_instances("krl", build(tmp_path, words, candidates))
    assert result.funnel[7].retained == 2
    assert [inst.word_id for inst in result.instances] == [100, 102]
    expected = [
        (
            100,
            10,
            1,
            "tere",
            '<s id="10"><w id="1">one</w></s>',
            (Candidate(9001, "SG+ACC"), Candidate(9001, "SG+NOM"), Candidate(9002, "SG+GEN")),
            Candidate(9001, "SG+NOM"),
        ),
        (
            102,
            12,
            1,
            "kolmas",
            '<s id="12"><w id="3">three</w></s>',
            (Candidate(9011, "SG+ACC"), Candidate(9011, "SG+NOM"), Candidate(9012, "SG+GEN")),
            Candidate(9011, "SG+NOM"),
        ),
    ]
    for instance, (word_id, sentence_id, text_id, word, xml, candidates_expected, gold) in zip(
        result.instances, expected
    ):
        assert instance.word_id == word_id
        assert instance.sentence_id == sentence_id
        assert instance.text_id == text_id
        assert instance.word == word
        assert instance.sentence_xml == xml
        assert instance.candidates == candidates_expected
        assert instance.gold_analysis == gold


def test_determine_data_tag_requires_git_checkout(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    with pytest.raises(CorpusTagError, match="dictorpus-data checkout tag"):
        determine_data_tag(repo)


@pytest.mark.skipif(shutil.which("git") is None, reason="Git unavailable")
def test_determine_data_tag_requires_single_tag(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "README.md").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        [
            "git", "-C", str(repo),
            "-c", "user.name=Test", "-c", "user.email=test@example.org",
            "commit", "-qm", "x",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "tag", "v1"], check=True)
    subprocess.run(["git", "-C", str(repo), "tag", "v2"], check=True)
    with pytest.raises(CorpusTagError, match="Multiple Git tags"):
        determine_data_tag(repo)
    subprocess.run(["git", "-C", str(repo), "tag", "-d", "v2"], check=True)
    assert determine_data_tag(repo) == "v1"


@pytest.mark.skipif(shutil.which("git") is None, reason="Git unavailable")
def test_cli_build_instances_against_tagged_checkout(tmp_path):
    for lang, sid in (("krl", 20), ("lud", 7), ("olo", 5), ("vep", 1)):
        write_corpus(
            tmp_path,
            lang,
            TEXTS,
            [(sid, 1, f'<s id="{sid}">teksti</s>', "")],
            [(1, sid, 0, "zero"), (2, sid, 1, "ok")],
            [(2, 9901, "SG+NOM", 2), (2, 9902, "SG+GEN", 1)],
        )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        [
            "git", "-C", str(tmp_path),
            "-c", "user.name=Test", "-c", "user.email=test@example.org",
            "commit", "-qm", "fixture",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "tag", "v2026.09"], check=True)
    out_dir = tmp_path / "derived" / "quality"
    result = _run_cli(
        "build-instances", "--data-dir", str(tmp_path), "--output-dir", str(out_dir)
    )
    assert result.returncode == 0, result.stderr
    path = out_dir / "zero_word_number_sentences_v2026.09.csv"
    assert path.is_file()
    with open(path, encoding="utf-8", newline="") as fh:
        lines = list(csv.reader(fh))
    assert lines[0] == ["language", "sentence_id"]
    assert lines[1:] == [["krl", "20"], ["lud", "7"], ["olo", "5"], ["vep", "1"]]
    corpus_files = sorted(p.name for p in (tmp_path / "corpus").iterdir())
    assert len(corpus_files) == 16


def test_cli_build_instances_fails_without_tag(tmp_path):
    build_fixture(tmp_path)
    result = _run_cli("build-instances", "--data-dir", str(tmp_path))
    assert result.returncode != 0
    assert "tag" in result.stderr.lower()