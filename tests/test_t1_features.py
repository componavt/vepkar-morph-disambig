import json
import sys
from dataclasses import FrozenInstanceError, replace
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
    _resolve_context,
    build_candidate_examples,
)
from core.instances import Candidate, Instance  # noqa: E402


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


def make_instance(
    language, word_id, sentence_id, word_number, word, candidates=None, gold_analysis=None
):
    if candidates is None:
        candidates = (Candidate(1, "SG"), Candidate(2, "PL"))
    if gold_analysis is None:
        gold_analysis = Candidate(1, "SG")
    return Instance(
        language=language,
        word_id=word_id,
        sentence_id=sentence_id,
        text_id=1,
        word=word,
        word_number=word_number,
        sentence_xml="",
        candidates=candidates,
        gold_analysis=gold_analysis,
    )


def make_word(language, sentence_id, word_id, word_number, word):
    return ContextWord(language, sentence_id, word_id, word_number, word)


class TestResolveContext:
    def test_ordinary_neighbors_absent_from_targets(self):
        instances = [make_instance("krl", 317, 50, 5, "B")]
        words = [
            make_word("krl", 50, 901, 2, "A"),
            make_word("krl", 50, 317, 5, "B"),
            make_word("krl", 50, 840, 9, "C"),
        ]
        assert _resolve_context(instances, words) == {("krl", 317): ("A", "C")}

    def test_boundaries_first_last_single_word(self):
        instances = [
            make_instance("krl", 100, 1, 1, "a"),
            make_instance("krl", 200, 2, 2, "b"),
            make_instance("krl", 300, 3, 1, "c"),
        ]
        words = [
            make_word("krl", 1, 100, 1, "a"),
            make_word("krl", 1, 101, 2, "x"),
            make_word("krl", 2, 200, 2, "b"),
            make_word("krl", 2, 201, 1, "y"),
            make_word("krl", 3, 300, 1, "c"),
        ]
        result = _resolve_context(instances, words)
        assert result[("krl", 100)] == ("<BOS>", "x")
        assert result[("krl", 200)] == ("y", "<EOS>")
        assert result[("krl", 300)] == ("<BOS>", "<EOS>")

    def test_repeated_spellings_different_ids(self):
        instances = [make_instance("krl", 2, 10, 2, "in")]
        words = [
            make_word("krl", 10, 1, 1, "in"),
            make_word("krl", 10, 2, 2, "in"),
            make_word("krl", 10, 3, 3, "in"),
        ]
        assert _resolve_context(instances, words) == {("krl", 2): ("in", "in")}

    def test_nonconsecutive_positions_and_order_invariance(self):
        instances = [
            make_instance("krl", 901, 50, 2, "A"),
            make_instance("krl", 317, 50, 5, "B"),
            make_instance("krl", 840, 50, 9, "C"),
        ]
        words = [
            make_word("krl", 50, 840, 9, "C"),
            make_word("krl", 50, 317, 5, "B"),
            make_word("krl", 50, 901, 2, "A"),
            make_word("krl", 50, 55, 20, "D"),
        ]
        expected = {
            ("krl", 901): ("<BOS>", "B"),
            ("krl", 317): ("A", "C"),
            ("krl", 840): ("B", "D"),
        }
        assert _resolve_context(instances, words) == expected
        assert _resolve_context(list(reversed(instances)), list(reversed(words))) == expected
        result = _resolve_context(instances, words)
        assert list(result) == sorted(result)

    def test_sentence_and_language_isolation(self):
        instances = [
            make_instance("krl", 7, 50, 1, "a"),
            make_instance("vep", 7, 50, 2, "b"),
        ]
        words = [
            make_word("krl", 50, 7, 1, "a"),
            make_word("krl", 50, 8, 2, "x"),
            make_word("vep", 50, 7, 2, "b"),
            make_word("vep", 50, 6, 1, "y"),
        ]
        assert _resolve_context(instances, words) == {
            ("krl", 7): ("<BOS>", "x"),
            ("vep", 7): ("y", "<EOS>"),
        }

    def test_missing_target_id_in_other_sentence(self):
        instances = [make_instance("krl", 317, 50, 5, "B")]
        words = [
            make_word("krl", 50, 901, 2, "A"),
            make_word("krl", 50, 840, 9, "C"),
            make_word("krl", 60, 317, 5, "B"),
        ]
        with pytest.raises(FeatureConstructionError) as exc:
            _resolve_context(instances, words)
        message = str(exc.value)
        assert "krl" in message and "50" in message and "317" in message
        assert "missing" in message.lower()

    def test_unrelated_sentences_with_invalid_positions_and_duplicates_ignored(self):
        instances = [make_instance("krl", 317, 50, 5, "B")]
        words = [
            make_word("krl", 50, 901, 2, "A"),
            make_word("krl", 50, 317, 5, "B"),
            make_word("krl", 50, 840, 9, "C"),
            make_word("vep", 999, 1, 0, "bad"),
            make_word("vep", 999, 2, 1, "dup1"),
            make_word("vep", 999, 3, 1, "dup2"),
            make_word("vep", 999, 4, 2, "dupid"),
            make_word("vep", 999, 4, 3, "dupid"),
        ]
        assert _resolve_context(instances, words) == {("krl", 317): ("A", "C")}

    @pytest.mark.parametrize("bad_position", [0, -1])
    def test_nonpositive_position_on_nontarget_word_in_needed_sentence(self, bad_position):
        instances = [make_instance("krl", 317, 50, 5, "B")]
        words = [
            make_word("krl", 50, 901, bad_position, "A"),
            make_word("krl", 50, 317, 5, "B"),
        ]
        with pytest.raises(FeatureConstructionError) as exc:
            _resolve_context(instances, words)
        message = str(exc.value)
        assert "krl" in message and "50" in message and "317" in message
        assert "nonpositive" in message.lower()

    def test_duplicate_position_on_nontarget_words_in_needed_sentence(self):
        instances = [make_instance("krl", 317, 50, 5, "B")]
        words = [
            make_word("krl", 50, 901, 2, "A"),
            make_word("krl", 50, 902, 2, "A2"),
            make_word("krl", 50, 317, 5, "B"),
        ]
        with pytest.raises(FeatureConstructionError) as exc:
            _resolve_context(instances, words)
        message = str(exc.value)
        assert "krl" in message and "50" in message and "317" in message
        assert "duplicate" in message.lower()

    def test_duplicate_position_reports_first_repeated_original_position(self):
        instances = [make_instance("krl", 317, 50, 5, "B")]
        words = [
            make_word("krl", 50, 901, 2, "A"),
            make_word("krl", 50, 902, 5, "A2"),
            make_word("krl", 50, 903, 5, "A3"),
            make_word("krl", 50, 904, 2, "A4"),
            make_word("krl", 50, 317, 5, "B"),
        ]
        with pytest.raises(FeatureConstructionError) as exc:
            _resolve_context(instances, words)
        message = str(exc.value)
        assert "duplicate word_number 2" in message
        assert "duplicate word_number 5" not in message

    def test_missing_target_error(self):
        instances = [make_instance("krl", 317, 50, 5, "B")]
        words = [make_word("krl", 50, 901, 2, "A")]
        with pytest.raises(FeatureConstructionError) as exc:
            _resolve_context(instances, words)
        message = str(exc.value)
        assert "krl" in message and "50" in message and "317" in message
        assert "missing" in message.lower()

    def test_multiple_records_with_target_id_error(self):
        instances = [make_instance("krl", 317, 50, 5, "B")]
        words = [
            make_word("krl", 50, 317, 5, "B"),
            make_word("krl", 50, 317, 6, "B2"),
            make_word("krl", 50, 901, 2, "A"),
        ]
        with pytest.raises(FeatureConstructionError) as exc:
            _resolve_context(instances, words)
        message = str(exc.value)
        assert "krl" in message and "50" in message and "317" in message
        assert "exactly one" in message

    def test_target_position_mismatch_error(self):
        instances = [make_instance("krl", 317, 50, 5, "B")]
        words = [make_word("krl", 50, 317, 6, "B")]
        with pytest.raises(FeatureConstructionError) as exc:
            _resolve_context(instances, words)
        message = str(exc.value)
        assert "krl" in message and "50" in message and "317" in message
        assert "expected 5" in message and "got 6" in message

    def test_target_word_string_mismatch_error(self):
        instances = [make_instance("krl", 317, 50, 5, "B")]
        words = [make_word("krl", 50, 317, 5, "b")]
        with pytest.raises(FeatureConstructionError) as exc:
            _resolve_context(instances, words)
        message = str(exc.value)
        assert "krl" in message and "50" in message and "317" in message
        assert "expected 'B'" in message and "got 'b'" in message

    def test_generator_inputs(self):
        instances = (
            make_instance(language, word_id, sentence_id, word_number, word)
            for language, word_id, sentence_id, word_number, word in [
                ("krl", 317, 50, 5, "B"),
                ("vep", 700, 80, 3, "w"),
            ]
        )
        words = (
            make_word(language, sentence_id, word_id, word_number, word)
            for language, sentence_id, word_id, word_number, word in [
                ("krl", 50, 901, 2, "A"),
                ("krl", 50, 317, 5, "B"),
                ("krl", 50, 840, 9, "C"),
                ("vep", 80, 700, 3, "w"),
                ("vep", 80, 601, 2, "v"),
                ("vep", 80, 750, 4, "x"),
            ]
        )
        assert _resolve_context(instances, words) == {
            ("krl", 317): ("A", "C"),
            ("vep", 700): ("v", "x"),
        }

    def test_input_objects_and_order_unchanged(self):
        instance_list = [
            make_instance("krl", 317, 50, 5, "B"),
            make_instance("krl", 901, 50, 2, "A"),
        ]
        word_list = [
            make_word("krl", 50, 901, 2, "A"),
            make_word("krl", 50, 840, 9, "C"),
            make_word("krl", 50, 317, 5, "B"),
        ]
        original_instances = list(instance_list)
        original_words = list(word_list)
        _resolve_context(instance_list, word_list)
        assert instance_list == original_instances
        assert word_list == original_words

    def test_empty_instances_skip_context(self):
        def raising_context():
            raise AssertionError("context_words must not be consumed")
            yield  # pragma: no cover

        assert _resolve_context([], raising_context()) == {}
        assert _resolve_context(iter([]), raising_context()) == {}

    def test_literal_preservation(self):
        instances = [make_instance("krl", 2, 10, 2, " MiXeD ")]
        words = [
            make_word("krl", 10, 1, 1, "  a  "),
            make_word("krl", 10, 2, 2, " MiXeD "),
            make_word("krl", 10, 3, 3, "e\u0301\u0301"),
        ]
        result = _resolve_context(instances, words)
        assert result[("krl", 2)] == ("  a  ", "e\u0301\u0301")


class GuardedInstance:
    _ALLOWED = frozenset({"language", "sentence_id", "word_id", "word_number", "word"})

    def __init__(self, instance):
        self._instance = instance

    def __getattr__(self, name):
        if name in self._ALLOWED:
            return getattr(self._instance, name)
        raise AssertionError(f"forbidden attribute access: {name}")


class TestGuardedContextAccess:
    def test_resolve_context_uses_only_allowed_fields(self):
        guarded = GuardedInstance(make_instance("krl", 317, 50, 5, "B"))
        words = [
            make_word("krl", 50, 901, 2, "A"),
            make_word("krl", 50, 317, 5, "B"),
            make_word("krl", 50, 840, 9, "C"),
        ]
        assert _resolve_context([guarded], words) == {("krl", 317): ("A", "C")}

    @pytest.mark.parametrize(
        "forbidden", ["candidates", "gold_analysis", "sentence_xml", "text_id"]
    )
    def test_forbidden_fields_raise_assertion_error(self, forbidden):
        guarded = GuardedInstance(make_instance("krl", 317, 50, 5, "B"))
        with pytest.raises(AssertionError):
            getattr(guarded, forbidden)


class NoGoldGuard:
    def __init__(self, instance):
        self._instance = instance

    def __getattr__(self, name):
        if name == "gold_analysis":
            raise AssertionError("gold_analysis must not be accessed")
        return getattr(self._instance, name)


class AssemblyGuard:
    _FORBIDDEN = frozenset({"sentence_xml", "text_id"})

    def __init__(self, instance):
        self._instance = instance

    def __getattr__(self, name):
        if name in self._FORBIDDEN:
            raise AssertionError(f"forbidden attribute access: {name}")
        return getattr(self._instance, name)


def make_context_words(language, sentence_id, triples):
    return [make_word(language, sentence_id, word_id, number, word) for word_id, number, word in triples]


class TestBuildCandidateExamples:
    def test_hand_built_assembly(self):
        instance = make_instance(
            "krl", 5, 50, 2, "in",
            candidates=(Candidate(7, "SG+GEN"),),
            gold_analysis=Candidate(7, "SG+GEN"),
        )
        words = make_context_words("krl", 50, [(1, 1, "A"), (5, 2, "in"), (9, 3, "C")])
        examples = build_candidate_examples([instance], words, mode="train")
        assert len(examples) == 1
        example = examples[0]
        assert example.language == "krl"
        assert example.word_id == 5
        assert example.wordform_id == 7
        assert example.gramset == "SG+GEN"
        assert example.word == "in"
        assert example.word_length == 2
        assert example.left_word == "A"
        assert example.right_word == "C"
        assert example.label == 1
        assert example.sample_weight == 1.0
        expected_features = {
            '["element","GEN"]': 1.0,
            '["element","SG"]': 1.0,
            '["gramset","SG+GEN"]': 1.0,
            '["language","krl"]': 1.0,
            '["language_gramset","krl","SG+GEN"]': 1.0,
            '["left_gramset","SG+GEN","A"]': 1.0,
            '["right_gramset","SG+GEN","C"]': 1.0,
            '["suffix_gramset","SG+GEN","1","n"]': 1.0,
            '["suffix_gramset","SG+GEN","2","in"]': 1.0,
        }
        assert example.features == expected_features

    def test_two_field_gold_matching_shared_id_different_gramset(self):
        three_candidate = make_instance(
            "krl", 1, 10, 2, "a",
            candidates=(Candidate(1, "SG"), Candidate(1, "PL"), Candidate(2, "SG")),
            gold_analysis=Candidate(1, "SG"),
        )
        two_candidate = make_instance(
            "krl", 2, 20, 2, "b",
            candidates=(Candidate(4, "SG"), Candidate(4, "PL")),
            gold_analysis=Candidate(4, "SG"),
        )
        words = (
            make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
            + make_context_words("krl", 20, [(2, 2, "b"), (3, 1, "x"), (4, 3, "y")])
        )
        examples = build_candidate_examples(
            [three_candidate, two_candidate], words, mode="train"
        )
        groups = {}
        for example in examples:
            groups.setdefault((example.language, example.word_id), []).append(example)
        assert list(groups) == [("krl", 1), ("krl", 2)]
        three_examples = groups[("krl", 1)]
        two_examples = groups[("krl", 2)]
        assert len(three_examples) == 3
        assert [e.sample_weight for e in three_examples] == pytest.approx(
            [1 / 3, 1 / 3, 1 / 3]
        )
        assert sum(e.sample_weight for e in three_examples) == pytest.approx(1.0)
        assert len(two_examples) == 2
        assert [e.sample_weight for e in two_examples] == pytest.approx(
            [1 / 2, 1 / 2]
        )
        assert sum(e.sample_weight for e in two_examples) == pytest.approx(1.0)
        by_key = {(e.wordform_id, e.gramset): e for e in examples}
        assert by_key[(1, "SG")].label == 1
        assert by_key[(1, "PL")].label == 0
        assert by_key[(2, "SG")].label == 0
        assert by_key[(4, "SG")].label == 1
        assert by_key[(4, "PL")].label == 0
        for group in groups.values():
            positives = [e for e in group if e.label == 1]
            assert len(positives) == 1

    def test_two_field_gold_matching_shared_gramset_different_id(self):
        instances = [
            make_instance(
                "krl", 1, 10, 2, "a",
                candidates=(Candidate(1, "SG"), Candidate(2, "SG")),
                gold_analysis=Candidate(2, "SG"),
            ),
            make_instance(
                "krl", 2, 20, 2, "b",
                candidates=(Candidate(3, "SG"), Candidate(4, "PL"), Candidate(5, "PL")),
                gold_analysis=Candidate(5, "PL"),
            ),
        ]
        words = make_context_words(
            "krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")]
        ) + make_context_words("krl", 20, [(2, 2, "b"), (3, 1, "x"), (4, 3, "y")])
        examples = build_candidate_examples(instances, words, mode="train")
        by_key = {(e.wordform_id, e.gramset): e for e in examples}
        assert by_key[(1, "SG")].label == 0
        assert by_key[(2, "SG")].label == 1
        assert by_key[(3, "SG")].label == 0
        assert by_key[(4, "PL")].label == 0
        assert by_key[(5, "PL")].label == 1

        grouped: dict[tuple[str, int], list[CandidateExample]] = {}
        for example in examples:
            grouped.setdefault((example.language, example.word_id), []).append(example)
        assert set(grouped) == {("krl", 1), ("krl", 2)}

        two_examples = grouped[("krl", 1)]
        assert len(two_examples) == 2
        assert [e.sample_weight for e in two_examples] == pytest.approx(
            [1 / 2, 1 / 2]
        )
        assert sum(e.sample_weight for e in two_examples) == pytest.approx(1.0)
        positives = [e for e in two_examples if e.label == 1]
        assert len(positives) == 1

        three_examples = grouped[("krl", 2)]
        assert len(three_examples) == 3
        assert [e.sample_weight for e in three_examples] == pytest.approx(
            [1 / 3, 1 / 3, 1 / 3]
        )
        assert sum(e.sample_weight for e in three_examples) == pytest.approx(1.0)
        positives = [e for e in three_examples if e.label == 1]
        assert len(positives) == 1

    def test_missing_gold_match_error(self):
        instance = make_instance(
            "krl", 1, 10, 2, "a",
            candidates=(Candidate(1, "SG"), Candidate(2, "PL")),
            gold_analysis=Candidate(9, "XX"),
        )
        words = make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
        with pytest.raises(FeatureConstructionError) as exc:
            build_candidate_examples([instance], words, mode="train")
        message = str(exc.value)
        assert "krl" in message and "10" in message and "1" in message
        assert "gold" in message.lower()

    def test_multiple_gold_matches_error(self):
        instance = make_instance(
            "krl", 1, 10, 2, "a",
            candidates=(Candidate(1, "SG"), Candidate(1, "SG")),
            gold_analysis=Candidate(1, "SG"),
        )
        words = make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
        with pytest.raises(FeatureConstructionError) as exc:
            build_candidate_examples([instance], words, mode="train")
        message = str(exc.value)
        assert "krl" in message and "10" in message and "1" in message
        assert "gold" in message.lower()

    def test_empty_candidates_train_error(self):
        instance = make_instance(
            "krl", 1, 10, 2, "a",
            candidates=(),
            gold_analysis=Candidate(1, "SG"),
        )
        words = make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
        with pytest.raises(FeatureConstructionError) as exc:
            build_candidate_examples([instance], words, mode="train")
        message = str(exc.value)
        assert "krl" in message and "10" in message and "1" in message
        assert "gold" in message.lower()

    def test_predict_labels_and_weights_none(self):
        instance = make_instance("krl", 1, 10, 2, "a")
        words = make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
        examples = build_candidate_examples([instance], words, mode="predict")
        assert len(examples) == 2
        assert all(e.label is None and e.sample_weight is None for e in examples)

    def test_predict_unchanged_after_gold_replacement(self):
        base = make_instance(
            "krl", 1, 10, 2, "a",
            candidates=(Candidate(1, "SG"), Candidate(2, "PL")),
            gold_analysis=Candidate(1, "SG"),
        )
        replaced = make_instance(
            "krl", 1, 10, 2, "a",
            candidates=(Candidate(1, "SG"), Candidate(2, "PL")),
            gold_analysis=Candidate(2, "PL"),
        )
        words = make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
        first = build_candidate_examples([base], words, mode="predict")
        second = build_candidate_examples([replaced], words, mode="predict")
        assert first == second

    def test_predict_guard_forbids_only_gold_access(self):
        instance = make_instance("krl", 1, 10, 2, "a")
        words = make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
        examples = build_candidate_examples([NoGoldGuard(instance)], words, mode="predict")
        assert len(examples) == 2
        assert all(e.label is None and e.sample_weight is None for e in examples)

    def test_replacing_train_gold_changes_only_labels(self):
        instance = make_instance(
            "krl", 1, 10, 2, "a",
            candidates=(Candidate(1, "SG"), Candidate(2, "PL")),
            gold_analysis=Candidate(1, "SG"),
        )
        words = make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
        first = build_candidate_examples([instance], words, mode="train")
        second = build_candidate_examples(
            [replace(instance, gold_analysis=Candidate(2, "PL"))],
            words,
            mode="train",
        )
        assert [e.label for e in first] == [1, 0]
        assert [e.label for e in second] == [0, 1]
        assert tuple(replace(e, label=None) for e in first) == tuple(
            replace(e, label=None) for e in second
        )

    def test_distinct_candidates_equal_features_and_isolated_dicts(self):
        instance = make_instance(
            "krl", 1, 10, 2, "a",
            candidates=(Candidate(1, "SG"), Candidate(2, "SG")),
            gold_analysis=Candidate(1, "SG"),
        )
        words = make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
        examples = build_candidate_examples([instance], words, mode="train")
        assert len(examples) == 2
        assert examples[0].features == examples[1].features
        assert examples[0].features is not examples[1].features
        examples[0].features.clear()
        assert examples[1].features

    @pytest.mark.parametrize("mode", ["train", "predict"])
    def test_exact_tuple_ordering_after_permutations(self, mode):
        base_instances = [
            make_instance("krl", 1, 10, 2, "a", candidates=(Candidate(1, "SG"), Candidate(2, "PL")), gold_analysis=Candidate(1, "SG")),
            make_instance("krl", 2, 20, 2, "b", candidates=(Candidate(3, "SG"), Candidate(4, "PL")), gold_analysis=Candidate(3, "SG")),
            make_instance("krl", 5, 50, 2, "e", candidates=(Candidate(7, "SG"), Candidate(7, "PL")), gold_analysis=Candidate(7, "SG")),
            make_instance("vep", 3, 30, 2, "c", candidates=(Candidate(5, "SG"), Candidate(6, "PL")), gold_analysis=Candidate(5, "SG")),
        ]
        base_words = make_context_words(
            "krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")]
        ) + make_context_words("krl", 20, [(2, 2, "b"), (3, 1, "x"), (4, 3, "y")]) + make_context_words("krl", 50, [(5, 2, "e"), (4, 1, "x"), (6, 3, "y")]) + make_context_words("vep", 30, [(3, 2, "c"), (4, 1, "x"), (5, 3, "y")])
        expected = build_candidate_examples(base_instances, base_words, mode=mode)
        expected_keys = [
            ("krl", 1, 1, "SG"),
            ("krl", 1, 2, "PL"),
            ("krl", 2, 3, "SG"),
            ("krl", 2, 4, "PL"),
            ("krl", 5, 7, "PL"),
            ("krl", 5, 7, "SG"),
            ("vep", 3, 5, "SG"),
            ("vep", 3, 6, "PL"),
        ]
        keys = [
            (e.language, e.word_id, e.wordform_id, e.gramset)
            for e in expected
        ]
        assert keys == expected_keys

        shuffled_candidates = [
            replace(instance, candidates=tuple(reversed(instance.candidates)))
            for instance in base_instances
        ]
        got = build_candidate_examples(
            list(reversed(shuffled_candidates)),
            list(reversed(base_words)),
            mode=mode,
        )
        assert got == expected
        got_keys = [
            (e.language, e.word_id, e.wordform_id, e.gramset)
            for e in got
        ]
        assert got_keys == expected_keys

    @pytest.mark.parametrize("mode", ["train", "predict"])
    def test_assembly_does_not_read_xml_or_text_id(self, mode):
        instance = make_instance(
            "krl", 1, 10, 2, "a",
            candidates=(Candidate(1, "SG"), Candidate(2, "PL")),
            gold_analysis=Candidate(1, "SG"),
        )
        words = make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
        expected = build_candidate_examples([instance], words, mode=mode)
        guarded = AssemblyGuard(instance)
        actual = build_candidate_examples([guarded], words, mode=mode)
        assert actual
        assert actual == expected

    def test_generator_inputs(self):
        instance_gen = (
            make_instance("krl", 1, 10, 2, "a")
            for _ in (0,)
        )
        word_gen = (
            word
            for word in make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
        )
        examples = build_candidate_examples(instance_gen, word_gen, mode="train")
        assert len(examples) == 2

    def test_empty_instances_without_consuming_context(self):
        def raising_context():
            raise AssertionError("context_words must not be consumed")
            yield  # pragma: no cover

        assert build_candidate_examples([], raising_context(), mode="train") == ()
        assert build_candidate_examples(iter([]), raising_context(), mode="predict") == ()

    def test_invalid_mode_rejected_before_consuming_input(self):
        def raising_instances():
            raise AssertionError("instances must not be consumed")
            yield  # pragma: no cover

        def raising_context():
            raise AssertionError("context_words must not be consumed")
            yield  # pragma: no cover

        with pytest.raises(FeatureConstructionError):
            build_candidate_examples(raising_instances(), raising_context(), mode="rank")

    def test_input_objects_unchanged(self):
        instance_list = [
            make_instance("krl", 1, 10, 2, "a"),
            make_instance("krl", 2, 20, 2, "b"),
        ]
        word_list = make_context_words(
            "krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")]
        ) + make_context_words("krl", 20, [(2, 2, "b"), (3, 1, "x"), (4, 3, "y")])
        original_instances = list(instance_list)
        original_words = list(word_list)
        build_candidate_examples(instance_list, word_list, mode="train")
        assert instance_list == original_instances
        assert word_list == original_words

    def test_train_predict_metadata_and_features_match(self):
        instance = make_instance(
            "krl", 1, 10, 2, "a",
            candidates=(Candidate(1, "SG"), Candidate(2, "PL")),
            gold_analysis=Candidate(1, "SG"),
        )
        words = make_context_words("krl", 10, [(1, 2, "a"), (2, 1, "x"), (3, 3, "y")])
        train = build_candidate_examples([instance], words, mode="train")
        predict = build_candidate_examples([instance], words, mode="predict")
        assert len(train) == len(predict)
        for te, pe in zip(train, predict):
            assert te.language == pe.language
            assert te.word_id == pe.word_id
            assert te.wordform_id == pe.wordform_id
            assert te.gramset == pe.gramset
            assert te.word == pe.word
            assert te.word_length == pe.word_length
            assert te.left_word == pe.left_word
            assert te.right_word == pe.right_word
            assert te.features == pe.features

    def test_context_errors_propagate(self):
        instance = make_instance("krl", 317, 50, 5, "B")
        words = [make_word("krl", 50, 901, 2, "A")]
        with pytest.raises(FeatureConstructionError) as exc:
            build_candidate_examples([instance], words, mode="train")
        message = str(exc.value)
        assert "missing" in message.lower()
