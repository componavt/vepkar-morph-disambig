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
    @pytest.mark.parametrize(
        ("gramset", "expected_elements"),
        [
            ("", {""}),
            ("SG++GEN", {"SG", "", "GEN"}),
            ("SG+SG", {"SG"}),
            ("PFV.PTCP+SG", {"PFV.PTCP", "SG"}),
        ],
    )
    def test_element_set_from_split(self, gramset, expected_elements):
        actual = _build_features(
            language="en",
            gramset=gramset,
            word="cat",
            left_word="a",
            right_word="b",
        )
        decoded = {tuple(json.loads(key)) for key in actual}
        assert {parts for parts in decoded if parts[0] == "element"} == {
            ("element", element) for element in expected_elements
        }
        assert ("gramset", gramset) in decoded
        for key in actual:
            parts = json.loads(key)
            if parts[0] == "element":
                assert type(actual[key]) is float and actual[key] == 1.0


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

    def test_four_character_word_limit(self):
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

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            (
                "häntä",
                {
                    ("suffix_gramset", "SG", "1", "ä"),
                    ("suffix_gramset", "SG", "2", "tä"),
                    ("suffix_gramset", "SG", "3", "ntä"),
                    ("suffix_gramset", "SG", "4", "äntä"),
                },
            ),
            (
                "a\u0308",
                {
                    ("suffix_gramset", "SG", "1", "\u0308"),
                    ("suffix_gramset", "SG", "2", "a\u0308"),
                },
            ),
        ],
    )
    def test_unicode_characters_preserved(self, word, expected):
        actual = _build_features(
            language="fi",
            gramset="SG",
            word=word,
            left_word="a",
            right_word="b",
        )
        suffix_keys = set()
        for key in actual:
            parts = json.loads(key)
            if parts[0] == "suffix_gramset":
                suffix_keys.add(tuple(parts))
        assert suffix_keys == expected


class TestJsonRoundTripping:
    @pytest.mark.parametrize(
        ("language", "gramset", "word", "left_word", "right_word"),
        [
            ("l1", 'A"B', 'C"D', 'E"F', 'G"H'),
            ("l2", "A\\B", "C\\D", "E\\F", "G\\H"),
            ("l3", 'A,B', "C:D", "E;F", 'G"'),
            ("l4", '"', "\\", ",", ":"),
            ("l5", 'A"B,C\\D', 'E"F,', 'G\\H,', 'I:J'),
            ("  MiXeD  ", "  SG +Nom ", "  Cat ", "  LEFT ", "  RIGHT "),
        ],
    )
    def test_round_trip_keys(self, language, gramset, word, left_word, right_word):
        actual = _build_features(
            language=language,
            gramset=gramset,
            word=word,
            left_word=left_word,
            right_word=right_word,
        )
        decoded = {tuple(json.loads(key)) for key in actual}
        assert ("language", language) in decoded
        assert ("gramset", gramset) in decoded
        assert ("language_gramset", language, gramset) in decoded
        assert ("left_gramset", gramset, left_word) in decoded
        assert ("right_gramset", gramset, right_word) in decoded
        assert {("element", element) for element in gramset.split("+")} == {
            parts for parts in decoded if parts[0] == "element"
        }
        assert {
            ("suffix_gramset", gramset, str(n), word[-n:])
            for n in range(1, min(4, len(word)) + 1)
        } == {parts for parts in decoded if parts[0] == "suffix_gramset"}
        for key in actual:
            parts = json.loads(key)
            assert isinstance(parts, list)
            assert all(isinstance(p, str) for p in parts)
            assert json.dumps(parts, ensure_ascii=False, separators=(",", ":")) == key

    def test_structural_collision_distinguished(self):
        first = _build_features(
            language="a",
            gramset="b,c",
            word="cat",
            left_word="a",
            right_word="b",
        )
        second = _build_features(
            language="a,b",
            gramset="c",
            word="cat",
            left_word="a",
            right_word="b",
        )
        first_keys = [
            key for key in first if json.loads(key)[0] == "language_gramset"
        ]
        second_keys = [
            key for key in second if json.loads(key)[0] == "language_gramset"
        ]
        assert len(first_keys) == 1
        assert len(second_keys) == 1
        first_key = first_keys[0]
        second_key = second_keys[0]
        assert json.loads(first_key) == ["language_gramset", "a", "b,c"]
        assert json.loads(second_key) == ["language_gramset", "a,b", "c"]
        assert first_key != second_key


class TestFeatureForms:
    def test_only_permitted_families(self):
        actual = _build_features(
            language="en",
            gramset="SG+N.PL",
            word="cats",
            left_word="<BOS>",
            right_word="<EOS>",
        )
        expected_sizes = {
            "gramset": 2,
            "element": 2,
            "language": 2,
            "language_gramset": 3,
            "suffix_gramset": 4,
            "left_gramset": 3,
            "right_gramset": 3,
        }
        for key in actual:
            parts = json.loads(key)
            assert isinstance(parts, list)
            assert parts
            assert all(isinstance(p, str) for p in parts)
            assert parts[0] in expected_sizes
            assert len(parts) == expected_sizes[parts[0]]

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
        assert actual
        assert all(type(v) is float and v == 1.0 for v in actual.values())

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
        expected_sizes = {
            "gramset": 2,
            "element": 2,
            "language": 2,
            "language_gramset": 3,
            "suffix_gramset": 4,
            "left_gramset": 3,
            "right_gramset": 3,
        }
        for key in actual:
            parts = json.loads(key)
            assert isinstance(parts, list)
            assert parts[0] in expected_sizes
            assert len(parts) == expected_sizes[parts[0]]