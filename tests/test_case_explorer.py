"""Tests for the case-explorer ground-truth mask rendering and payload shape."""

import numpy as np

from conceptlab.case_explorer import render_gt_mask


FIELDS = ["amount", "dt", "cvv", "country"]


def test_decision_scope():
    M = render_gt_mask([("cvv", "decision")], T=6, fields=FIELDS)
    assert M.sum() == 1 and M[5, 2] == 1


def test_window_scope():
    M = render_gt_mask([("dt", "window:3")], T=6, fields=FIELDS)
    assert M[:, 1].tolist() == [0, 0, 0, 1, 1, 1]
    assert M.sum() == 3


def test_history_scope():
    M = render_gt_mask([("country", "history")], T=4, fields=FIELDS)
    assert M[:, 3].tolist() == [1, 1, 1, 0]


def test_combined_scopes():
    M = render_gt_mask([("country", "decision"), ("country", "history")], T=4, fields=FIELDS)
    assert M[:, 3].tolist() == [1, 1, 1, 1]
    assert M[:, :3].sum() == 0


def test_unknown_field_ignored():
    M = render_gt_mask([("nonexistent", "decision")], T=4, fields=FIELDS)
    assert M.sum() == 0


def test_window_larger_than_T():
    M = render_gt_mask([("dt", "window:99")], T=4, fields=FIELDS)
    assert M[:, 1].sum() == 4
