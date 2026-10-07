import argparse
import math

import pytest
import torch

from scsi_new.train_utils import atomic_save, check_args, make_lr_lambda, rng_state, set_rng_state


def test_cosine_endpoints_and_clamp():
    f = make_lr_lambda("cosine", warmup_steps=10, mstep_steps=5, horizon_scsi_steps=2, floor=0.1)
    assert f(0) == pytest.approx(1.0)
    assert f(20) == pytest.approx(0.1)              # T = 10 + 2 * 5
    assert f(500) == pytest.approx(0.1)             # clamped, not re-warmed past the horizon
    assert f(10) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.5)))


def test_constant_and_start_step():
    assert make_lr_lambda("constant", 10, 5, 2, 0.1)(7) == 1.0
    f = make_lr_lambda("cosine", 10, 5, 2, 0.1, start_step=20)
    assert f(0) == pytest.approx(0.1)               # resumed at the end of the schedule


def test_cosine_per_mstep_restarts():
    f = make_lr_lambda("cosine_per_mstep", warmup_steps=10, mstep_steps=4, horizon_scsi_steps=3, floor=0.0)
    assert f(0) == pytest.approx(1.0) and f(10) == pytest.approx(1.0) and f(14) == pytest.approx(1.0)
    assert f(12) == pytest.approx(0.5)


def test_unknown_schedule():
    with pytest.raises(ValueError):
        make_lr_lambda("nope", 1, 1, 1, 0.0)(0)


def test_check_args():
    args = argparse.Namespace(a=1, b=[1, 2], c="x")
    check_args({"a": 1, "b": (1, 2)}, args, ("a", "b"), "p")          # tuple vs list is equal
    with pytest.raises(ValueError, match="different args"):
        check_args({"a": 2, "b": [1, 2]}, args, ("a", "b"), "p")
    check_args({"a": 1}, argparse.Namespace(a=1, c="x"), ("a", "c"), "p", arg_defaults={"c": "x"})
    with pytest.raises(ValueError):                                     # absent key = old default
        check_args({"a": 1}, argparse.Namespace(a=1, c="y"), ("a", "c"), "p", arg_defaults={"c": "x"})


def test_rng_roundtrip():
    torch.manual_seed(0)
    state = rng_state()
    a = torch.randn(4)
    set_rng_state(state)
    assert torch.equal(a, torch.randn(4))


def test_atomic_save(tmp_path):
    p = tmp_path / "sub" / "latest.pt"
    atomic_save({"x": torch.arange(3)}, str(p))
    assert torch.equal(torch.load(p)["x"], torch.arange(3))
    assert not (tmp_path / "sub" / "latest.pt.tmp").exists()
