"""T35: bit-identity check of our port against the teammate notebook's own definitions (revised notebook, cell 28).

Needs the private notebook copy (never in Git).  Only the function definitions and the listed constants are extracted
by AST (no data loading, splitting, model fitting or plotting code) and executed on frames handed in by the caller
(synthetic in unit tests, fold-0 rows in the real-data preflight).  This module contains none of the notebook's code.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.ey_pipeline_v2 import features as F

CELL_INDEX = 28
NEEDED_FUNCTIONS = ['building_features', 'rate_table', 'make_features', 'to_matrix', 'make_weights']
NEEDED_ASSIGNS = ['SEED', 'CAT_COLS', 'RATE_KEYS', 'NEW_FEATURES', 'ALL_FEATURES']


def cell_text(notebook_path, index=CELL_INDEX):
    nb = json.loads(Path(notebook_path).read_text(encoding='utf-8'))
    cell = nb['cells'][index]
    F.require(cell['cell_type'] == 'code', 'Cell is not a code cell')
    return ''.join(cell['source'])


def extract(cell_source):
    """Parse the cell; return (namespace with the extracted definitions, extraction evidence)."""
    tree = ast.parse(cell_source)
    assigns, funcs, imports = {}, {}, set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in NEEDED_ASSIGNS:
                F.require(name not in assigns, f'Duplicate assignment: {name}')
                assigns[name] = node
        elif isinstance(node, ast.FunctionDef) and node.name in NEEDED_FUNCTIONS:
            F.require(node.name not in funcs, f'Duplicate definition: {node.name}')
            funcs[node.name] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.update(a.asname or a.name for a in node.names)
    F.require(set(assigns) == set(NEEDED_ASSIGNS), f'Missing assignments: {set(NEEDED_ASSIGNS) - set(assigns)}')
    F.require(set(funcs) == set(NEEDED_FUNCTIONS), f'Missing functions: {set(NEEDED_FUNCTIONS) - set(funcs)}')
    F.require({'np', 'pd', 'KFold'} <= imports, 'Expected imports not found in the cell')
    F.require(ast.literal_eval(assigns['SEED'].value) == F.SEED, 'SEED differs')
    F.require(ast.literal_eval(assigns['CAT_COLS'].value) == F.TEXT_CAT, 'CAT_COLS differs')
    F.require(ast.literal_eval(assigns['RATE_KEYS'].value) == F.RATE_KEYS, 'RATE_KEYS differs')
    body = sorted(list(assigns.values()) + list(funcs.values()), key=lambda n: n.lineno)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    text = ast.unparse(module)
    ns = {'np': np, 'pd': pd, 'KFold': KFold, 'ORIGINAL': F.original_columns()}
    exec(compile(module, '<notebook-cell-28-extract>', 'exec'), ns)  # trusted private copy, SHA-pinned by the caller
    F.require(list(ns['ALL_FEATURES']) == F.feature_columns(), 'Notebook ALL_FEATURES differs from our 75-column order')
    evidence = dict(functions=[n.name for n in body if isinstance(n, ast.FunctionDef)],
                    assignments=[n.targets[0].id for n in body if isinstance(n, ast.Assign)],
                    supplied=['np', 'pd', 'KFold', 'ORIGINAL (our 38 raw inputs; equality with ALL_FEATURES prefix checked)'],
                    extract_text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
    return ns, evidence


def frames_identical(a, b):
    """Exact equality including column order, dtypes and NaN positions."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, 'columns or length differ'
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True,
                                      check_dtype=True)
    except AssertionError as exc:
        return False, str(exc).splitlines()[0][:300]
    return True, ''


def compare(ns, train_df, valid_df, strength=0.5):
    """Our fold construction versus the notebook definitions on the same frames (bit-identical required)."""
    valid_df = valid_df.drop(columns=[F.GRADE], errors='ignore')
    ours_X_tr, ours_X_va, ours_tr, ours_va, _ = F.fold_matrices(train_df, valid_df)
    o_base, (o_valid,) = ns['make_features'](train_df, [valid_df])
    cols = list(ns['ALL_FEATURES'])
    o_X_tr, o_X_va = ns['to_matrix']([o_base, o_valid], cols)
    checks = {}
    for name, a, b in [('features_train', ours_tr, o_base[cols]), ('features_valid', ours_va, o_valid[cols]),
                       ('matrix_train', ours_X_tr, o_X_tr), ('matrix_valid', ours_X_va, o_X_va)]:
        ok, why = frames_identical(a, b)
        checks[name] = dict(identical=ok, reason=why)
    y0 = pd.Series(train_df[F.GRADE].to_numpy() - 1)
    w_ours, w_nb = F.class_weights(y0, strength), ns['make_weights'](y0, strength)
    checks['weights'] = dict(identical=bool(np.array_equal(w_ours, w_nb)))
    return dict(checks=checks, matrix_columns=int(ours_X_tr.shape[1]),
                passes=bool(all(v['identical'] for v in checks.values())))
