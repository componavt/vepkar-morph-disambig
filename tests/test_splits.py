"""Offline tests for whole-text weighted splits and the shared split CSV."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.instances import Candidate, Instance  # noqa: E402
from core.splits import (  # noqa: E402
    SPLIT_NAMES,
    SplitError,
    TextWeight,
    assign_texts,
    build_split_rows,
    compute_text_weights,
    design_text_weights,
    split_csv_bytes,
    split_csv_path,
    write_split_csv,
)

TAG = "v2026.09"


def _candidates(count: int = 2) -> tuple[Candidate, ...]:
    return tuple(Candidate(index, f"G{index}") for index in range(1, count + 1))


def _instance(
    language: str,
    word_id: int,
    text_id: int,
    sentence_id: int,
    candidate_count: int = 2,
) -> Instance:
    candidates = _candidates(candidate_count)
    return Instance(
        language=language,
        word_id=word_id,
        sentence_id=sentence_id,
        text_id=text_id,
        word=f"w{word_id}",
        word_number=1,
        sentence_xml="<s/>",
        candidates=candidates,
        gold_analysis=candidates[0],
    )


def _text_instances(
    language: str,
    text_id: int,
    word_start: int,
    word_count: int,
) -> list[Instance]:
    return [
        _instance(language, word_start + index, text_id, text_id * 100 + index)
        for index in range(word_count)
    ]


def test_per_text_weights_count_instances_and_candidates():
    instances = (
        _text_instances("krl", 1, 1000, 3)
        + [_instance("krl", 1110, 1, 110, candidate_count=3)]
        + _text_instances("krl", 2, 2000, 2)
    )
    weights = compute_text_weights(instances)
    assert weights == (
        TextWeight("krl", 1, 4, 9),
        TextWeight("krl", 2, 2, 4),
    )


def test_per_text_weight_is_candidate_sum_not_instance_count():
    instances = _text_instances("krl", 1, 100, 3) + _text_instances("krl", 2, 200, 5)
    weights = compute_text_weights(instances)
    by_text = {weight.text_id: weight for weight in weights}
    assert by_text[1].candidate_row_count == 6
    assert by_text[2].candidate_row_count == 10
    assert by_text[1].candidate_row_count == sum(
        len(inst.candidates) for inst in instances if inst.text_id == 1
    )


def test_text_id_across_languages_rejected():
    instances = _text_instances("krl", 7, 1000, 2) + _text_instances("olo", 7, 2000, 2)
    with pytest.raises(SplitError, match="text_id=7"):
        compute_text_weights(instances)


def test_same_input_identical_assignment_and_csv_bytes():
    weights = design_text_weights(
        [(lang, text_id, 2, 4) for lang in ("vep", "krl", "olo", "lud") for text_id in range(1, 9)]
    )
    first = {
        lang: assign_texts(weights) for lang in ("vep", "krl", "olo", "lud")
    }
    second = {
        lang: assign_texts(weights) for lang in ("vep", "krl", "olo", "lud")
    }
    assert first == second
    assert split_csv_bytes(build_split_rows(first)) == split_csv_bytes(
        build_split_rows(second)
    )


def test_attainable_eighty_ten_ten_on_candidate_rows():
    for lang in ("vep", "krl", "olo", "lud"):
        weights = design_text_weights(
            [(lang, text_id, 5, 10) for text_id in range(1, 11)]
        )
        assignment = assign_texts(weights)
        rows = {
            part: sum(w.candidate_row_count for w in assignment[part])
            for part in SPLIT_NAMES
        }
        assert rows == {"train": 80, "dev": 10, "test": 10}


def test_unattainable_reports_actual_proportions():
    weights = design_text_weights([("krl", 1, 50, 100)])
    assignment = assign_texts(weights)
    rows = {
        part: sum(w.candidate_row_count for w in assignment[part])
        for part in SPLIT_NAMES
    }
    assert rows["train"] + rows["dev"] + rows["test"] == 100
    assert rows != {"train": 80, "dev": 10, "test": 10}
    assert rows["train"] == 100


def test_every_split_nonempty_for_sufficient_fixture():
    weights = design_text_weights(
        [("krl", text_id, count, count * 2) for text_id, count in enumerate((5, 5, 3, 3, 2, 2), start=1)]
    )
    assignment = assign_texts(weights)
    for part in SPLIT_NAMES:
        assert assignment[part]
        assert sum(w.instance_count for w in assignment[part]) > 0


def test_text_with_many_instances_belongs_to_one_split():
    instances = _text_instances("krl", 3, 300, 7)
    weights = compute_text_weights(instances)
    assignment = assign_texts(weights)
    parts_of_text = [
        part for part in SPLIT_NAMES if any(w.text_id == 3 for w in assignment[part])
    ]
    assert len(parts_of_text) == 1


def test_each_instance_inherits_exactly_one_split():
    instances = _text_instances("krl", 1, 100, 4) + _text_instances("krl", 2, 200, 4)
    weights = compute_text_weights(instances)
    assignment = assign_texts(weights)
    split_of_text = {
        w.text_id: part for part in SPLIT_NAMES for w in assignment[part]
    }
    assert set(split_of_text) == {1, 2}
    for instance in instances:
        assert split_of_text[instance.text_id] in SPLIT_NAMES


def _probe_rows() -> tuple[tuple[str, int, str], ...]:
    return build_split_rows(
        {
            "krl": {
                "train": (TextWeight("krl", 1, 2, 4),),
                "dev": (TextWeight("krl", 2, 1, 2),),
                "test": (TextWeight("krl", 3, 1, 2),),
            }
        }
    )


def test_existing_identical_split_csv_preserved(tmp_path):
    target = split_csv_path(tmp_path, TAG)
    content = split_csv_bytes(_probe_rows())
    assert write_split_csv(target, content) is True
    assert write_split_csv(target, content) is False
    assert target.read_bytes() == content


def test_different_existing_split_csv_rejected(tmp_path):
    target = split_csv_path(tmp_path, TAG)
    original = split_csv_bytes(_probe_rows())
    content = original.decode("utf-8")
    altered = content.replace("\n", ",\n").encode("utf-8")
    assert altered != original
    write_split_csv(target, original)
    with pytest.raises(SplitError, match="not be overwritten"):
        write_split_csv(target, altered)
    assert target.read_bytes() == original


def test_split_csv_schema_unique_sorted_values(tmp_path):
    rows = build_split_rows(
        {
            lang: assign_texts(
                design_text_weights([(lang, text_id, 2, 4) for text_id in (5, 1, 3, 9)])
            )
            for lang in ("vep", "krl", "olo", "lud")
        }
    )
    content = split_csv_bytes(rows)
    lines = list(csv.reader(content.decode("utf-8").splitlines()))
    assert lines[0] == ["language", "text_id", "split"]
    data = lines[1:]
    assert len(data) == len({tuple(row) for row in data})
    keys = [(lang, int(tid)) for lang, tid, _ in data]
    assert keys == sorted(keys)
    assert all(split in {"train", "dev", "test"} for _, _, split in data)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SRC / "cli.py"), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _write_zst_corpus(root: Path, lang: str, rows) -> None:
    import io

    import pandas as pd
    import zstandard as zstd

    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for name, columns, data in rows:
        frame = pd.DataFrame.from_records(data, columns=columns)
        csv_text = frame.to_csv(index=False, lineterminator="\n")
        cctx = zstd.ZstdCompressor(level=3)
        with open(corpus / f"{name}_{lang}.csv.zst", "wb") as fh:
            cctx.copy_stream(io.BytesIO(csv_text.encode("utf-8")), fh)


TEXT_COLS = ["text_id", "corpus_id", "dialect_code", "genre_id", "year_recorded"]
SENT_COLS = ["sentence_id", "text_id", "sentence_xml", "sentence_ru"]
WORD_COLS = ["word_id", "sentence_id", "word_number", "word"]
CAND_COLS = ["word_id", "wordform_id", "gramset", "relevance"]


def build_language_fixture(root: Path, lang: str, offset: int) -> None:
    base = offset * 100
    texts = [(base + index, "", "", "", "") for index in range(1, 7)]
    sentences = []
    words = []
    candidates = []
    word_id = offset
    sentence_id = offset * 1000
    wordform = offset * 100
    for index, word_count in enumerate((5, 5, 3, 3, 2, 2), start=1):
        text_id = base + index
        for _ in range(word_count):
            sentences.append(
                (sentence_id, text_id, f'<s id="{sentence_id}"><w id="{word_id}">sana</w></s>', "")
            )
            words.append((word_id, sentence_id, 1, "sana"))
            candidates.append((word_id, wordform, "N+SG+NOM", 2))
            candidates.append((word_id, wordform + 1, "N+SG+GEN", 1))
            word_id += 1
            wordform += 2
            sentence_id += 1
    _write_zst_corpus(
        root,
        lang,
        (
            ("texts", TEXT_COLS, texts),
            ("sentences", SENT_COLS, sentences),
            ("words", WORD_COLS, words),
            ("candidate_analyses", CAND_COLS, candidates),
        ),
    )


def _tagged_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    for lang, offset in (("vep", 1), ("krl", 101), ("olo", 201), ("lud", 301)):
        build_language_fixture(checkout, lang, offset)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "-A"], check=True)
    subprocess.run(
        [
            "git", "-C", str(checkout),
            "-c", "user.name=Test", "-c", "user.email=test@example.org",
            "commit", "-qm", "fixture",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "tag", TAG], check=True)
    return checkout


GIT_AVAILABLE = subprocess.run(
    ["git", "--version"], capture_output=True
).returncode == 0


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_cli_make_splits_full_run(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    out_dir = tmp_path / "derived"
    result = _run_cli(
        "make-splits", "--data-dir", str(checkout), "--output-dir", str(out_dir)
    )
    assert result.returncode == 0, result.stderr
    for lang in ("vep", "krl", "olo", "lud"):
        assert f"  {lang}  OK" in result.stderr
    assert "BENCHMARK — dictorpus-data v2026.09" in result.stdout
    assert (
        "SPLIT — target by candidate rows: train 80% / dev 10% / test 10%"
        in result.stdout
    )
    assert "Checks: text overlap = 0; instance overlap = 0" in result.stdout
    assert f"Split: {out_dir / 'splits_v2026.09.csv'}" in result.stdout
    assert "Δ" not in result.stdout
    target = split_csv_path(out_dir, TAG)
    assert target.is_file()
    first_lines = target.read_text(encoding="utf-8").splitlines()
    assert first_lines[0] == "language,text_id,split"
    assert len(first_lines) == 25  # header + 6 texts x 4 languages
    second = _run_cli(
        "make-splits", "--data-dir", str(checkout), "--output-dir", str(out_dir)
    )
    assert second.returncode == 0, second.stderr
    assert target.read_text(encoding="utf-8").splitlines() == first_lines


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
def test_cli_build_instances_compact_output(tmp_path):
    checkout = _tagged_checkout(tmp_path)
    out_dir = tmp_path / "derived" / "quality"
    result = _run_cli(
        "build-instances", "--data-dir", str(checkout), "--output-dir", str(out_dir)
    )
    assert result.returncode == 0, result.stderr
    assert "FUNNEL" not in result.stdout
    assert "Source tag:" not in result.stdout
    assert "BENCHMARK — dictorpus-data v2026.09" in result.stdout
    assert "Empty-candidate loss" in result.stdout
    for lang in ("vep", "krl", "olo", "lud"):
        assert f"  {lang}  OK" in result.stderr
    review = out_dir / "zero_word_number_sentences_v2026.09.csv"
    assert review.is_file()
    corpus_files = sorted(p.name for p in (checkout / "corpus").iterdir())
    assert len(corpus_files) == 16


def test_cli_help_has_no_verbose_flag():
    result = _run_cli("--help")
    assert "--verbose" not in result.stdout
    help_splits = _run_cli("make-splits", "--help")
    assert "--verbose" not in help_splits.stdout
    assert "Δ" not in help_splits.stdout