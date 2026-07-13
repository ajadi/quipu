"""Unit tests for quipu.embeddings.engine.

All tests use injected fake session/tokenizer — no model file required.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import pytest
import numpy as np
from tests._semantic import TEST_EMBED_DIM

from quipu.embeddings.engine import (
    _Engine,
    _reset,
    embed,
    embed_batch,
    embed_dim,
    set_engine,
)
from quipu.models.cache import active_model


@pytest.mark.parametrize("model", [None, "none"])
def test_import_is_safe_in_keyword_only_mode(model):
    env = os.environ.copy()
    if model is None:
        env.pop("QUIPU_EMBEDDING_MODEL", None)
    else:
        env["QUIPU_EMBEDDING_MODEL"] = model
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from quipu.embeddings import EMBED_DIM, embed_dim\n"
            "assert EMBED_DIM is None\n"
            "try:\n"
            "    embed_dim()\n"
            "except ValueError:\n"
            "    print('safe')\n"
            "else:\n"
            "    raise AssertionError('embed_dim unexpectedly succeeded')\n",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "safe"


def test_keyword_only_in_process_contract(monkeypatch):
    monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "none")

    assert active_model() is None
    with pytest.raises(ValueError, match="keyword-only"):
        embed_dim()


# ---------------------------------------------------------------------------
# Fake session / tokenizer helpers
# ---------------------------------------------------------------------------

class _FakeTokenizerEncoding:
    def __init__(self, ids, mask):
        self.ids = ids
        self.attention_mask = mask


class _FakeTokenizer:
    """Minimal stub matching the tokenizers.Tokenizer interface used by encode_batch."""

    def __init__(self, seq_len: int = 8) -> None:
        self._seq_len = seq_len

    def encode_batch(self, texts):
        return [
            _FakeTokenizerEncoding(
                ids=[1] * self._seq_len,
                mask=[1] * self._seq_len,
            )
            for _ in texts
        ]


class _DistinctFakeTokenizer:
    """Tokenizer that emits DISTINCT ids per input text.

    The first token id is derived from the text so that different inputs
    produce different id vectors. This makes per-row output distinguishable
    in the session, allowing order and name-based resolution bugs to surface.
    """

    def __init__(self, seq_len: int = 8) -> None:
        self._seq_len = seq_len

    def encode_batch(self, texts):
        encodings = []
        for text in texts:
            # Use a stable hash of the text to produce a unique first id in
            # range [2, 999] (avoids 0=pad, 1=uniform default).
            first_id = (hash(text) % 998) + 2
            ids = [first_id] + [1] * (self._seq_len - 1)
            mask = [1] * self._seq_len
            encodings.append(_FakeTokenizerEncoding(ids=ids, mask=mask))
        return encodings


class _FakeSession:
    """Returns deterministic float32 output of a configurable rank/shape."""

    def __init__(
        self,
        output_rank: int = 3,
        seq_len: int = 8,
        value: float = 1.0,
        input_order: tuple = ("input_ids", "attention_mask"),
        extra_inputs: tuple = (),
    ) -> None:
        self._rank = output_rank
        self._seq_len = seq_len
        self._value = value
        self._input_order = input_order
        self._extra_inputs = extra_inputs

    def get_inputs(self):
        names = list(self._input_order) + list(self._extra_inputs)
        return [_N(name) for name in names]

    def get_outputs(self):
        return [_N("last_hidden_state" if self._rank == 3 else "sentence_embedding")]

    def run(self, output_names, feeds):
        n = feeds["input_ids"].shape[0]
        if self._rank == 3:
            arr = np.full((n, self._seq_len, TEST_EMBED_DIM), self._value, dtype=np.float32)
        else:
            arr = np.full((n, TEST_EMBED_DIM), self._value, dtype=np.float32)
        return [arr]


class _DistinctFakeSession:
    """Session whose output varies per-row AND per-dimension.

    Each row's embedding derives from ``input_ids[row, :]``:

        output[row, dim] = input_ids[row, dim % seq_len] * (dim + 1)

    This means:
    - Different per-row id sequences → different raw vectors (distinct texts).
    - Raw norm varies per row (non-trivial, non-unit).
    - Different input arrays (ids vs mask) produce detectably different
      directions after L2-normalization, because values at each sequence
      position tile into the 384 dimensions (not just scale the same ramp).

    Name-reversal test: feeding the attention_mask (all-ones) as input_ids
    produces ``output[row, dim] = 1 * (dim+1) = ramp`` for every row.
    Feeding the true ids (first token = first_id ≠ 1, rest = 1) produces a
    tiled pattern that differs across dimensions mod seq_len, giving a
    different unit vector — detectable.

    For rank-3 every token position carries the same per-row row-vector so
    that mean-pooling over the sequence axis recovers exactly that row-vector.
    """

    def __init__(
        self,
        output_rank: int = 3,
        seq_len: int = 8,
        input_order: tuple = ("input_ids", "attention_mask"),
    ) -> None:
        self._rank = output_rank
        self._seq_len = seq_len
        self._input_order = input_order

    def get_inputs(self):
        return [_N(name) for name in self._input_order]

    def get_outputs(self):
        return [_N("last_hidden_state" if self._rank == 3 else "sentence_embedding")]

    def run(self, output_names, feeds):
        ids = feeds["input_ids"]  # shape (N, seq_len), dtype int64
        n = ids.shape[0]
        seq_len = ids.shape[1]
        # dim_ramp: shape (EMBED_DIM,) = [1, 2, ..., 384]
        dim_ramp = np.arange(1, TEST_EMBED_DIM + 1, dtype=np.float32)
        # For each embedding dim d, the token index that contributes is d % seq_len.
        # tile_ids[row, d] = ids[row, d % seq_len]
        dim_indices = np.arange(TEST_EMBED_DIM) % seq_len  # (TEST_EMBED_DIM,)
        tile_ids = ids[:, dim_indices].astype(np.float32)  # (N, TEST_EMBED_DIM)
        # pooled[row, dim] = tile_ids[row, dim] * dim_ramp[dim]
        pooled = tile_ids * dim_ramp[np.newaxis, :]  # (N, TEST_EMBED_DIM)
        if self._rank == 2:
            return [pooled]
        # Rank-3: broadcast pooled row across all token positions so that
        # mean-pooling recovers pooled exactly (each token == same row-vector).
        arr = np.broadcast_to(
            pooled[:, np.newaxis, :], (n, self._seq_len, TEST_EMBED_DIM)
        ).copy()
        return [arr]


class _N:
    """Tiny name-holder for session inputs/outputs."""
    def __init__(self, name: str) -> None:
        self.name = name
        # Default type string — engine checks for "int" substring.
        self.type = "tensor(int64)"


def _distinct_expected_vec(text: str, seq_len: int = 8) -> list:
    """Compute the expected unit vector that _DistinctFakeSession + _DistinctFakeTokenizer
    would produce for *text*.

    Mirrors the formula in _DistinctFakeSession.run:
        output[row, dim] = ids[row, dim % seq_len] * (dim + 1)

    where ids[row] = [first_id, 1, 1, ..., 1] from _DistinctFakeTokenizer.
    Returns the L2-normalized float list of length EMBED_DIM.
    """
    first_id = (hash(text) % 998) + 2
    ids = np.array([first_id] + [1] * (seq_len - 1), dtype=np.float32)
    dim_ramp = np.arange(1, TEST_EMBED_DIM + 1, dtype=np.float32)
    dim_indices = np.arange(TEST_EMBED_DIM) % seq_len
    tile_ids = ids[dim_indices]             # (TEST_EMBED_DIM,)
    raw = tile_ids * dim_ramp              # (TEST_EMBED_DIM,)
    return (raw / np.linalg.norm(raw)).tolist()


def _make_engine(rank: int = 3, seq_len: int = 8, value: float = 1.0) -> _Engine:
    return _Engine(
        session=_FakeSession(output_rank=rank, seq_len=seq_len, value=value),
        tokenizer=_FakeTokenizer(seq_len=seq_len),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure singleton is clean before and after each test."""
    _reset()
    yield
    _reset()


@pytest.fixture()
def fake_engine_rank3():
    e = _make_engine(rank=3)
    set_engine(e)
    return e


@pytest.fixture()
def fake_engine_rank2():
    e = _make_engine(rank=2)
    set_engine(e)
    return e


# ---------------------------------------------------------------------------
# Tests: single embed
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_returns_correct_dimension(self, fake_engine_rank3):
        result = embed("hello world")
        assert len(result) == TEST_EMBED_DIM

    def test_returns_list_of_float(self, fake_engine_rank3):
        result = embed("hello world")
        assert all(isinstance(v, float) for v in result)

    def test_l2_normalized(self):
        """Normalization must divide by the actual L2 norm, not a constant.

        _DistinctFakeSession emits a per-dimension value computed as
        ids[row, dim % seq_len] * (dim + 1), which varies across dimensions
        (not a uniform vector).  The raw L2 norm is NOT 1 and NOT
        sqrt(384)*const.  After normalization each output[d] must equal
        raw[d] / ||raw||.  A normalizer that divided by a hardcoded value
        would produce different per-dimension values and fail the element-wise
        check.
        """
        text = "hello world"
        engine = _Engine(
            session=_DistinctFakeSession(output_rank=3, seq_len=8),
            tokenizer=_DistinctFakeTokenizer(seq_len=8),
        )
        set_engine(engine)

        expected_vec = _distinct_expected_vec(text, seq_len=8)
        result = embed(text)
        for got, exp in zip(result, expected_vec):
            assert abs(got - exp) < 1e-5, (
                "Element-wise check failed — normalizer may be dividing by "
                "the wrong constant rather than the actual L2 norm"
            )

    def test_rank2_passthrough_normalized(self):
        """Rank-2 pass-through must L2-normalize with a per-dimension-varying raw vector.

        Uses _DistinctFakeSession (rank-2) which emits ids[row, dim%seq_len]*(dim+1).
        The raw norm differs from 1.0 so a no-op normalizer would leave the
        result non-unit and the element-wise check would fail.
        """
        text = "test rank-2 output"
        engine = _Engine(
            session=_DistinctFakeSession(output_rank=2, seq_len=8),
            tokenizer=_DistinctFakeTokenizer(seq_len=8),
        )
        set_engine(engine)

        expected_vec = _distinct_expected_vec(text, seq_len=8)
        result = embed(text)
        assert len(result) == TEST_EMBED_DIM
        for got, exp in zip(result, expected_vec):
            assert abs(got - exp) < 1e-5, (
                "Rank-2 passthrough must L2-normalize the raw vector element-wise"
            )


# ---------------------------------------------------------------------------
# Tests: batch embed
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    def test_empty_returns_empty(self):
        # No engine needed — early return before singleton access.
        assert embed_batch([]) == []

    def test_batch_length_preserved(self, fake_engine_rank3):
        texts = ["one", "two", "three"]
        result = embed_batch(texts)
        assert len(result) == 3

    def test_each_vector_correct_dim(self, fake_engine_rank3):
        result = embed_batch(["a", "b"])
        for vec in result:
            assert len(vec) == TEST_EMBED_DIM

    def test_each_vector_normalized(self, fake_engine_rank3):
        result = embed_batch(["a", "b", "c"])
        for vec in result:
            norm = math.sqrt(sum(v * v for v in vec))
            assert abs(norm - 1.0) < 1e-5

    def test_order_preserved(self):
        """Each output vector must correspond to its own input text, not a neighbour.

        _DistinctFakeTokenizer gives each text a unique first token id.
        _DistinctFakeSession scales the output ramp by that id, so the
        resulting unit vector for text[i] has a unique direction.  If
        encode() returned rows in the wrong order the per-text id-to-vector
        mapping would break and the assertions below would fail.
        """
        texts = ["alpha", "beta", "gamma"]
        engine = _Engine(
            session=_DistinctFakeSession(output_rank=3, seq_len=8),
            tokenizer=_DistinctFakeTokenizer(seq_len=8),
        )
        set_engine(engine)
        result = embed_batch(texts)

        # Build what each individual embed() call produces (single-text path,
        # same tokenizer/session, so same first_id → same unit vector).
        expected = [
            embed_batch([t])[0]
            for t in texts
        ]

        # Each result[i] must match the vector for texts[i], not any other.
        for i, (got, exp) in enumerate(zip(result, expected)):
            for g, e in zip(got, exp):
                assert abs(g - e) < 1e-5, (
                    f"result[{i}] does not match embed(texts[{i}]); "
                    "order is wrong or a neighbour's vector was returned"
                )

    def test_single_item_batch(self, fake_engine_rank3):
        result = embed_batch(["only one"])
        assert len(result) == 1
        assert len(result[0]) == TEST_EMBED_DIM


# ---------------------------------------------------------------------------
# Tests: truncation (no-raise)
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_long_input_does_not_raise(self):
        """Tokenizer with tiny seq_len truncates silently."""
        # seq_len=4 simulates a very short max_length.
        e = _make_engine(rank=3, seq_len=4)
        set_engine(e)
        long_text = "word " * 10_000
        # Must not raise.
        result = embed(long_text)
        assert len(result) == TEST_EMBED_DIM


# ---------------------------------------------------------------------------
# Tests: pooling branches
# ---------------------------------------------------------------------------

class TestPoolingBranches:
    def test_rank3_mean_pool(self):
        """Rank-3 path: mean-pool over token axis, then L2-normalize.

        _DistinctFakeSession (rank-3) sets every token position in a row to
        the same per-row vector (ids[row, dim%seq_len] * (dim+1)).
        Mean-pooling over the token axis therefore recovers that row-vector
        exactly, and L2-normalization produces a known unit vector.  A buggy
        pooling implementation (e.g. summing without dividing by token count,
        or pooling the wrong axis) would produce a different raw value and the
        element-wise assertion would fail.
        """
        text = "rank-3 path"
        engine = _Engine(
            session=_DistinctFakeSession(output_rank=3, seq_len=8),
            tokenizer=_DistinctFakeTokenizer(seq_len=8),
        )
        set_engine(engine)

        expected_vec = _distinct_expected_vec(text, seq_len=8)
        result = embed(text)
        assert len(result) == TEST_EMBED_DIM
        for got, exp in zip(result, expected_vec):
            assert abs(got - exp) < 1e-5, (
                "Rank-3 mean-pool result does not match expected element-wise; "
                "pooling or normalization may be incorrect"
            )

    def test_rank2_passthrough(self):
        """Rank-2 path: pass-through (already pooled) then L2-normalize.

        Same element-wise verification as test_rank3_mean_pool but exercises
        the rank-2 branch (no pooling step).  A bug that re-pools rank-2
        output over a spurious axis would change the values and fail.
        """
        text = "rank-2 path"
        engine = _Engine(
            session=_DistinctFakeSession(output_rank=2, seq_len=8),
            tokenizer=_DistinctFakeTokenizer(seq_len=8),
        )
        set_engine(engine)

        expected_vec = _distinct_expected_vec(text, seq_len=8)
        result = embed(text)
        assert len(result) == TEST_EMBED_DIM
        for got, exp in zip(result, expected_vec):
            assert abs(got - exp) < 1e-5, (
                "Rank-2 passthrough result does not match expected element-wise"
            )


# ---------------------------------------------------------------------------
# Tests: pad-invariance
# ---------------------------------------------------------------------------

class TestPadInvariance:
    def test_masked_padding_does_not_shift_mean(self):
        """Pad tokens must not contribute to pooled vector — direction test.

        Real tokens point in an arithmetic-ramp direction [1, 2, ..., 384].
        Pad tokens point in the REVERSE direction [-384, -383, ..., -1], which
        is orthogonal in no dimension to the real direction.  If the attention
        mask is ignored and pad rows are included in the mean, the per-
        dimension signs partially cancel, producing a direction far from the
        ramp direction.  The test asserts element-wise equality against the
        mask-correct expected vector, so a mask-ignoring implementation would
        produce a different unit vector and fail.
        """
        ramp = np.arange(1, TEST_EMBED_DIM + 1, dtype=np.float32)         # [1..384]
        reverse = np.arange(-TEST_EMBED_DIM, 0, dtype=np.float32)          # [-384..-1]

        class _DirMixedMaskSession:
            def get_inputs(self):
                return [_N("input_ids"), _N("attention_mask")]

            def get_outputs(self):
                return [_N("last_hidden_state")]

            def run(self, output_names, feeds):
                # 1 sample, seq_len=4: 2 real tokens (ramp), 2 pad (reverse).
                arr = np.array([
                    [
                        ramp,    # token 0 — real
                        ramp,    # token 1 — real
                        reverse, # token 2 — pad (orthogonally different direction)
                        reverse, # token 3 — pad
                    ]
                ], dtype=np.float32)
                return [arr]

        class _DirMaskTokenizer:
            def encode_batch(self, texts):
                return [_FakeTokenizerEncoding(
                    ids=[1, 2, 0, 0],
                    mask=[1, 1, 0, 0],  # first 2 real, last 2 pad
                )]

        engine = _Engine(
            session=_DirMixedMaskSession(),
            tokenizer=_DirMaskTokenizer(),
        )
        set_engine(engine)
        result = embed("test pad invariance")

        # Expected: mean of real tokens = ramp, then L2-normalized.
        expected_raw = ramp  # mean of [ramp, ramp] = ramp
        expected_vec = (expected_raw / np.linalg.norm(expected_raw)).tolist()

        for d, (got, exp) in enumerate(zip(result, expected_vec)):
            assert abs(got - exp) < 1e-5, (
                f"dim {d}: got {got:.6f} expected {exp:.6f}; "
                "pad tokens may be included in the mean-pool (mask ignored)"
            )

        # Sanity: confirm the pad-included direction would be DIFFERENT.
        # mean of all 4 tokens (ignoring mask) = (ramp + ramp + reverse + reverse) / 4
        # = (2*ramp + 2*reverse) / 4 = (ramp + reverse) / 2
        padded_mean = (ramp + reverse) / 2.0
        norm_pm = np.linalg.norm(padded_mean)
        if norm_pm > 1e-9:
            wrong_vec = (padded_mean / norm_pm).tolist()
            # At least one dimension must differ materially from the correct result.
            max_diff = max(abs(g - w) for g, w in zip(expected_vec, wrong_vec))
            assert max_diff > 0.01, (
                "Pad and real tokens have the same direction — test is not discriminating"
            )


# ---------------------------------------------------------------------------
# Tests: name-based input resolution (Fix 2)
# ---------------------------------------------------------------------------

class TestNameBasedInputResolution:
    def test_reversed_input_order_still_works(self):
        """Engine must feed inputs BY NAME even when mask precedes ids in get_inputs().

        _DistinctFakeSession computes output[row, dim] = input_ids[row,0] * dim_ramp.
        _DistinctFakeTokenizer assigns a text-derived first_id > 1 for input_ids
        and a mask of all-ones (first value = 1).

        With CORRECT name-based resolution:
            feeds["input_ids"]      = [first_id, 1, 1, ...]  (distinct first token)
            feeds["attention_mask"] = [1, 1, 1, ...]         (all-ones mask)

        With a POSITIONAL bug (engine feeds positional slot 0 → attention_mask,
        slot 1 → input_ids, matching the reversed declaration order):
            feeds["input_ids"]      = [1, 1, 1, ...]  (the mask values!)
            feeds["attention_mask"] = [first_id, 1, ...]

        _DistinctFakeSession tiles ids[row, dim%seq_len] * (dim+1) across 384
        dimensions.  When ids[0]=first_id≠1, dims where (dim%8)==0 get value
        first_id*(dim+1) instead of 1*(dim+1).  This produces a detectably
        different raw vector (and different unit vector) from the all-ones mask
        scenario.  We assert element-wise equality to the name-correct expected
        vector, so a positional bug produces a different direction and fails.
        """
        text = "reversed order"
        seq_len = 8
        session = _DistinctFakeSession(
            output_rank=3,
            seq_len=seq_len,
            # Declaration order is reversed: mask slot listed before ids slot.
            input_order=("attention_mask", "input_ids"),
        )
        engine = _Engine(session=session, tokenizer=_DistinctFakeTokenizer(seq_len=seq_len))
        set_engine(engine)
        result = embed(text)

        assert len(result) == TEST_EMBED_DIM

        # Expected: name-correct vector uses first_id derived from text.
        expected_vec = _distinct_expected_vec(text, seq_len=seq_len)

        for d, (got, exp) in enumerate(zip(result, expected_vec)):
            assert abs(got - exp) < 1e-5, (
                f"dim {d}: got {got:.6f} expected {exp:.6f}; "
                "engine may be binding inputs positionally rather than by name"
            )

        # Sanity: confirm the positional-bug vector WOULD differ from the correct one.
        # When all input_ids = 1 (mask fed as ids), raw[dim] = 1 * (dim+1) = ramp.
        first_id = (hash(text) % 998) + 2
        assert first_id != 1, f"first_id={first_id} == 1 would make this test non-discriminating"
        dim_ramp = np.arange(1, TEST_EMBED_DIM + 1, dtype=np.float32)
        wrong_raw = dim_ramp  # all ids=1 → 1*(dim+1)
        wrong_vec = (wrong_raw / np.linalg.norm(wrong_raw)).tolist()
        max_diff = max(abs(e - w) for e, w in zip(expected_vec, wrong_vec))
        assert max_diff > 0.01, (
            "The positional-bug output is too similar to the correct output "
            "— the test is not discriminating for this text"
        )

    def test_extra_aux_input_does_not_raise(self):
        """Models with token_type_ids or position_ids must not crash — zeros fed."""
        session = _FakeSession(
            output_rank=3,
            seq_len=8,
            extra_inputs=("token_type_ids",),
        )
        engine = _Engine(session=session, tokenizer=_FakeTokenizer(seq_len=8))
        set_engine(engine)
        result = embed("aux input present")
        assert len(result) == TEST_EMBED_DIM

    def test_missing_input_ids_raises_runtime_error(self):
        """If input_ids cannot be found by name, RuntimeError must be raised."""
        class _BadSession:
            def get_inputs(self):
                return [_N("weird_input_a"), _N("weird_input_b")]

            def get_outputs(self):
                return [_N("last_hidden_state")]

            def run(self, output_names, feeds):  # pragma: no cover
                return [np.zeros((1, 8, TEST_EMBED_DIM), dtype=np.float32)]

        with pytest.raises(RuntimeError, match="input_ids"):
            _Engine(session=_BadSession(), tokenizer=_FakeTokenizer())

    def test_missing_attention_mask_raises_runtime_error(self):
        """If attention_mask cannot be found by name, RuntimeError must be raised."""
        class _NoMaskSession:
            def get_inputs(self):
                return [_N("input_ids"), _N("unrecognized_second")]

            def get_outputs(self):
                return [_N("last_hidden_state")]

            def run(self, output_names, feeds):  # pragma: no cover
                return [np.zeros((1, 8, TEST_EMBED_DIM), dtype=np.float32)]

        with pytest.raises(RuntimeError, match="attention"):
            _Engine(session=_NoMaskSession(), tokenizer=_FakeTokenizer())


# ---------------------------------------------------------------------------
# Tests: tokenizer max_seq_len cap (Fix 4)
# ---------------------------------------------------------------------------

class TestMaxSeqLenCap:
    def test_absurd_config_value_clamped(self, tmp_path):
        """An absurdly large model_max_length must not produce a value above 32768."""
        import json
        from quipu.embeddings.tokenizer import _resolve_max_seq_len, MAX_SEQ_LEN_CAP

        cfg = tmp_path / "tokenizer_config.json"
        cfg.write_text(json.dumps({"model_max_length": 10**18}), encoding="utf-8")

        result = _resolve_max_seq_len(tmp_path)
        assert result <= MAX_SEQ_LEN_CAP, (
            f"Expected result <= {MAX_SEQ_LEN_CAP}, got {result}"
        )

    def test_reasonable_value_passes_through(self, tmp_path):
        """A value within the cap should be used as-is."""
        import json
        from quipu.embeddings.tokenizer import _resolve_max_seq_len

        cfg = tmp_path / "tokenizer_config.json"
        cfg.write_text(json.dumps({"model_max_length": 512}), encoding="utf-8")

        assert _resolve_max_seq_len(tmp_path) == 512

    def test_value_exactly_at_cap_passes(self, tmp_path):
        """A value equal to the cap is valid."""
        import json
        from quipu.embeddings.tokenizer import _resolve_max_seq_len, MAX_SEQ_LEN_CAP

        cfg = tmp_path / "tokenizer_config.json"
        cfg.write_text(json.dumps({"model_max_length": MAX_SEQ_LEN_CAP}), encoding="utf-8")

        assert _resolve_max_seq_len(tmp_path) == MAX_SEQ_LEN_CAP

    def test_missing_config_returns_fallback(self, tmp_path):
        """Absent tokenizer_config.json returns the 2048 fallback."""
        from quipu.embeddings.tokenizer import _resolve_max_seq_len, MAX_SEQ_LEN_FALLBACK

        assert _resolve_max_seq_len(tmp_path) == MAX_SEQ_LEN_FALLBACK
