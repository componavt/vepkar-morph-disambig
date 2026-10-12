import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from t1_features.features import (  # noqa: E402
    CandidateExample,
    ContextWord,
    FeatureConstructionError,
    _build_features,
)


class TestTypes:
    def test_exception_inheritance(self):
        assert issubclass(FeatureConstructionError, ValueError)
        assert issubclass(FeatureConstructionError, Exception)
        assert FeatureConstructionError is not ValueError

    def test_context_word_fields(self):
        field_names = [f.name for f in ContextWord.__dataclass_fields__.values()]
        assert field_names == ["language", "sentence_id", "word_id", "word_number", "word"]
        annotations = ContextWord.__annotations__
        assert annotations["language"] is str
        assert annotations["sentence_id"] is int
        assert annotations["word_id"] is int
        assert annotations["word_number"] is int
        assert annotations["word"] is str

    def test_candidate_example_fields(self):
        field_names = [f.name for f in CandidateExample.__dataclass_fields__.values()]
        assert field_names == [
            "language",
            "word_id",
            "wordform_id",
            "gramset",
            "word",
            "word_length",
            "left_word",
            "right_word",
            "features",
            "label",
            "sample_weight",
        ]
        annotations = CandidateExample.__annotations__
        assert annotations["language"] is str
        assert annotations["word_id"] is int
        assert annotations["wordform_id"] is int
        assert annotations["gramset"] is str
        assert annotations["word"] is str
        assert annotations["word_length"] is int
        assert annotations["left_word"] is str
        assert annotations["right_word"] is str
        assert annotations["features"] == dict[str, float]
        assert annotations["label"] == int | None
        assert annotations["sample_weight"] == float | None

    def test_frozen_reassignment_blocked(self):
        cw = ContextWord("krl", 1, 2, 3, "in")
        with pytest.raises(FrozenInstanceError):
            cw.word = "out"
        ce = CandidateExample(
            "krl",
            1,
            2,
            "SG+GEN",
            "in",
            2,
            "<BOS>",
            "<EOS>",
            {},
            None,
            None,
        )
        with pytest.raises(FrozenInstanceError):
            ce.gramset = "PL+GEN"

    def test_nested_dict_remains_mutable(self):
        feats = {"[\"language\",\"en\"]": 1.0}
        ce = CandidateExample(
            "en", 1, 1, "SG", "x", 1, "<BOS>", "<EOS>", feats, 1, 0.5
        )
        ce.features["[\"language\",\"en\"]"] = 2.0
        ce.features["[\"gramset\",\"SG\"]"] = 1.0
        assert ce.features["[\"language\",\"en\"]"] == 2.0
        assert "[\"gramset\",\"SG\"]" in ce.features

    def test_candidate_example_constructs(self):
        ce = CandidateExample(
            "krl", 7, 9, "SG+GEN", "in", 2, "<BOS>", "<EOS>", {}, None, None
        )
        assert ce.language == "krl"
        assert ce.label is None
        assert ce.sample_weight is None


class TestHandBuiltCase:
    def test_expected_dictionary(self):
        actual = _build_features(
            language="krl",
            gramset="SG+GEN",
            word="in",
            left_word="<BOS>",
            right_word="<EOS>",
        )
        expected = {
            '["element","GEN"]': 1.0,
            '["element","SG"]': 1.0,
            '["gramset","SG+GEN"]': 1.0,
            '["language","krl"]': 1.0,
            '["language_gramset","krl","SG+GEN"]': 1.0,
            '["left_gramset","SG+GEN","<BOS>"]': 1.0,
            '["right_gramset","SG+GEN","<EOS>"]': 1.0,
            '["suffix_gramset","SG+GEN","1","n"]': 1.0,
            '["suffix_gramset","SG+GEN","2","in"]': 1.0,
        }
        assert actual == expected
        assert list(actual) == list(expected)


class TestGramsetElements:
    def test_repeated_elements_single_key(self):
        actual = _build_features(
            language="en",
            gramset="SG+SG+PL",
            word="cat",
            left_word="a",
            right_word="b",
        )
        element_keys = [k for k in actual if k.startswith('["element",')]
        assert element_keys == ['["element","PL"]', '["element","SG"]']

    def test_dotted_elements_unchanged(self):
        actual = _build_features(
            language="en",
            gramset="N.M.SG+NOM",
            word="cat",
            left_word="a",
            right_word="b",
        )
        assert '["element","N.M.SG"]' in actual
        assert '["element","NOM"]' in actual

    def test_empty_gramset(self):
        actual = _build_features(
            language="en",
            gramset="",
            word="cat",
            left_word="a",
            right_word="b",
        )
        assert "[\"element\"]" not in actual
        assert '["gramset",""]' in actual


class TestSuffixes:
    def test_ordinary_word(self):
        actual = _build_features(
            language="en",
            gramset="SG",
            word="cats",
            left_word="a",
            right_word="b",
        )
        assert '["suffix_gramset","SG","1","s"]' in actual
        assert '["suffix_gramset","SG","2","ts"]' in actual
        assert '["suffix_gramset","SG","3","ats"]' in actual
        assert '["suffix_gramset","SG","4","cats"]' in actual

    def test_short_suffix_equals_whole_word(self):
        actual = _build_features(
            language="en",
            gramset="SG",
            word="in",
            left_word="a",
            right_word="b",
        )
        assert '["suffix_gramset","SG","2","in"]' in actual
        assert "suffix_gramset" in "".join(actual)
        suffix_keys = [k for k in actual if '"suffix_gramset"' in k]
        assert len(suffix_keys) == 2

    def test_one_character_word(self):
        actual = _build_features(
            language="en",
            gramset="SG",
            word="i",
            left_word="a",
            right_word="b",
        )
        suffix_keys = [k for k in actual if '"suffix_gramset"' in k]
        assert suffix_keys == ['["suffix_gramset","SG","1","i"]']

    def test_empty_word_no_suffix_features(self):
        actual = _build_features(
            language="en",
            gramset="SG",
            word="",
            left_word="a",
            right_word="b",
        )
        assert not [k for k in actual if '"suffix_gramset"' in k]

    def test_unicode_word_four_character_limit(self):
        actual = _build_features(
            language="fi",
            gramset="SG",
            word="kissa",
            left_word="a",
            right_word="b",
        )
        suffix_keys = [k for k in actual if '"suffix_gramset"' in k]
        assert suffix_keys == [
            '["suffix_gramset","SG","1","a"]',
            '["suffix_gramset","SG","2","sa"]',
            '["suffix_gramset","SG","3","ssa"]',
            '["suffix_gramset","SG","4","issa"]',
        ]

    def test_unicode_characters_preserved(self):
        actual = _build_features(
            language="fi",
            gramset="SG",
            word="häntä",
            left_word="a",
            right_word="b",
        )
        assert '["suffix_gramset","SG","0"]' not in actual
        assert any("hä" in k or "ntä" in k for k in actual if '"suffix_gramset"' in k)


class TestJsonRoundTripping:
    @pytest.mark.parametrize(
        ("gramset", "word", "left_word", "right_word"),
        [
            ('A"B', 'C"D', 'E"F', 'G"H'),
            ("A\\B", "C\\D", "E\\F", "G\\H"),
            ('A,B', "C:D", "E;F", 'G"'),
            ('"', "\\", ",", ":"),
            ('A"B,C\\D', 'E"F,', 'G\\H,', 'I:J'),
        ],
    )
    def test_round_trip_keys(self, gramset, word, left_word, right_word):
        actual = _build_features(
            language="l",
            gramset=gramset,
            word=word,
            left_word=left_word,
            right_word=right_word,
        )
        for key in actual:
            parts = json.loads(key)
            assert isinstance(parts, list)
            assert all(isinstance(p, str) for p in parts)
            assert json.dumps(parts, ensure_ascii=False, separators=(",", ":")) == key

    def test_repr_values_are_distinct(self):
        actual = _build_features(
            language="en",
            gramset="SG",
            word="cat",
            left_word="a",
            right_word="b",
        )
        keys = list(actual)
        assert len(keys) == len(set(keys))


class TestFeatureForms:
    def test_only_permitted_families(self):
        actual = _build_features(
            language="en",
            gramset="SG+N.PL",
            word="cats",
            left_word="<BOS>",
            right_word="<EOS>",
        )
        allowed_prefixes = ("element", "gramset", "language", "language_gramset",
                            "suffix_gramset", "left_gramset", "right_gramset")
        for key in actual:
            parts = json.loads(key)
            assert parts[0] in allowed_prefixes

    @pytest.mark.parametrize(
        "parts",
        [
            ["gramset", "SG"],
            ["element", "SG"],
            ["language", "en"],
            ["language_gramset", "en", "SG"],
            ["suffix_gramset", "SG", "1", "t"],
            ["left_gramset", "SG", "<BOS>"],
            ["right_gramset", "SG", "<EOS>"],
        ],
    )
    def test_structural_form_present(self, parts):
        key = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
        actual = _build_features(
            language="en",
            gramset="SG",
            word="cat",
            left_word="<BOS>",
            right_word="<EOS>",
        )
        assert actual[key] == 1.0

    def test_dotted_element_not_split(self):
        actual = _build_features(
            language="en",
            gramset="N.PL",
            word="cat",
            left_word="a",
            right_word="b",
        )
        assert '["element","N.PL"]' in actual
        assert not any(json.loads(k)[0] == "element" and isinstance(json.loads(k)[1], str) and "." in json.loads(k)[1] and len(json.loads(k)) != 2 for k in actual)
        for key in actual:
            parts = json.loads(key)
            if parts[0] == "element":
                assert len(parts) == 2


class TestValuesAndOrder:
    def test_all_values_are_one(self):
        actual = _build_features(
            language="en",
            gramset="SG+NOM",
            word="cat",
            left_word="a",
            right_word="b",
        )
        assert set(actual.values()) == {1.0}

    def test_sorted_key_order(self):
        actual = _build_features(
            language="en",
            gramset="SG+NOM",
            word="cat",
            left_word="a",
            right_word="b",
        )
        assert list(actual) == sorted(actual)
        assert list(actual) == sorted(list(actual))

    def test_deterministic_across_calls(self):
        kwargs = dict(
            language="krl",
            gramset="SG+GEN",
            word="in",
            left_word="<BOS>",
            right_word="<EOS>",
        )
        first = _build_features(**kwargs)
        for _ in range(5):
            assert _build_features(**kwargs) == first
            assert list(_build_features(**kwargs)) == list(first)

    def test_independent_dictionaries(self):
        kwargs = dict(
            language="krl",
            gramset="SG+GEN",
            word="in",
            left_word="<BOS>",
            right_word="<EOS>",
        )
        a = _build_features(**kwargs)
        b = _build_features(**kwargs)
        assert a is not b
        a.clear()
        assert b == _build_features(**kwargs)

    def test_no_forbidden_feature_content(self):
        actual = _build_features(
            language="en",
            gramset="SG",
            word="cats",
            left_word="<BOS>",
            right_word="<EOS>",
        )
        joined = "".join(actual)
        assert "word_length" not in joined
        assert "length" not in json.dumps([json.loads(k) for k in actual], ensure_ascii=False)
        parsed = [json.loads(k) for k in actual]
        for parts in parsed:
            assert parts[0] in ("gramset", "element", "language", "language_gramset", "suffix_gramset", "left_gramset", "right_gramset")
            assert len(parts) <= 4