"""T33: bit-identity check of our port against the teammate notebook's own function definitions.

Needs the private notebook copy (never in Git).  Only the function definitions and the three numeric constants of
cell 25 are extracted by AST (no other globals, no plotting or model libraries) and executed on data frames that are
handed in by the caller (synthetic in unit tests, training-fold rows in the real-data preflight).  The module itself
contains none of the notebook's code.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from experiments.ey_pipeline_benchmark import features as F

NEEDED_FUNCTIONS = ['add_building_features', 'add_one_level', 'add_region_composition', 'damage_rate_table',
                    'add_target_rates', 'make_weights']
NEEDED_CONSTANTS = {'SEED': 42, 'MIN_BUILDINGS': 20, 'MIN_COMBO': 50}
REQUIRED_IMPORT_NAMES = {'pandas': 'pd', 'KFold': 'KFold'}  # the cell imports these; we supply the same objects


def cell_text(notebook_path, index=25):
    nb = json.loads(Path(notebook_path).read_text(encoding='utf-8'))
    cell = nb['cells'][index]
    F.require(cell['cell_type'] == 'code', 'Cell is not a code cell')
    return ''.join(cell['source'])


def extract(cell_source):
    """Parse the cell; return (namespace with the extracted definitions, dict of extraction evidence)."""
    tree = ast.parse(cell_source)
    consts, funcs, imports = {}, {}, set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in NEEDED_CONSTANTS:
                F.require(isinstance(node.value, ast.Constant) and node.value.value == NEEDED_CONSTANTS[name],
                          f'Constant {name} differs from the registered value')
                consts[name] = node
        elif isinstance(node, ast.FunctionDef) and node.name in NEEDED_FUNCTIONS:
            F.require(node.name not in funcs, f'Duplicate definition: {node.name}')
            funcs[node.name] = node
        elif isinstance(node, ast.Import):
            imports.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.update(a.asname or a.name for a in node.names)
    F.require(set(consts) == set(NEEDED_CONSTANTS), 'Missing constant definitions')
    F.require(set(funcs) == set(NEEDED_FUNCTIONS), 'Missing function definitions')
    F.require({'pd', 'KFold', 'train_test_split'} <= imports, 'Expected imports not found in the cell')
    body = sorted(list(consts.values()) + list(funcs.values()), key=lambda n: n.lineno)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    text = ast.unparse(module)
    ns = {'pd': pd, 'KFold': KFold, 'train_test_split': train_test_split}
    exec(compile(module, '<notebook-cell-25-extract>', 'exec'), ns)  # trusted private copy, SHA-pinned by the caller
    evidence = dict(functions=[n.name for n in body if isinstance(n, ast.FunctionDef)],
                    constants={k: NEEDED_CONSTANTS[k] for k in consts},
                    imports_supplied=sorted(['pd', 'KFold', 'train_test_split']),
                    extract_text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
    return ns, evidence


def run_original(ns, train_df, eval_df):
    """The notebook's order of operations (feature part only) with its own function objects."""
    tr = ns['add_building_features'](train_df.reset_index(drop=True))
    te = ns['add_building_features'](eval_df.drop(columns=[F.GRADE], errors='ignore').reset_index(drop=True))
    tr, te = ns['add_region_composition'](tr, te)
    tr, te = ns['add_target_rates'](tr, te, F.OLD_KEYS, min_count=50)
    tr, te = ns['add_target_rates'](tr, te, F.GEO_KEYS, min_count=10)
    tr, te = ns['add_target_rates'](tr, te, ['key_material'], min_count=50)
    for d in (tr, te):
        for lv in ('g2', 'g3'):
            d[f'{lv}_material_high_gap'] = d[f'{lv}_material_high_rate'] - d['material_high_rate']
            d[f'{lv}_material_low_gap'] = d[f'{lv}_material_low_rate'] - d['material_low_rate']
    for c in F.CAT_COLS:
        cats = tr[c].astype('category').cat.categories
        tr[c] = pd.Categorical(tr[c], categories=cats)
        te[c] = pd.Categorical(te[c], categories=cats)
    cols = F.feature_columns()
    return tr[cols], te[cols]


def frames_identical(a, b):
    """Exact equality including column order, dtypes (categorical vocabulary too) and NaN positions."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, 'columns or length differ'
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True,
                                      check_dtype=True, check_categorical=True)
    except AssertionError as exc:
        return False, str(exc).splitlines()[0][:300]
    return True, ''


def compare(ns, train_df, eval_df):
    """Our `build_features` versus the notebook functions on the same frames (bit-identical required)."""
    ours_tr, ours_te, _ = F.build_features(train_df, eval_df)
    orig_tr, orig_te = run_original(ns, train_df, eval_df)
    ok_tr, why_tr = frames_identical(ours_tr, orig_tr)
    ok_te, why_te = frames_identical(ours_te, orig_te)
    y0 = pd.Series(train_df[F.GRADE].to_numpy() - 1)
    weights_equal = bool((ns['make_weights'](y0, 0.3) == F.class_weights(y0, 0.3)).all())
    return dict(train_identical=ok_tr, eval_identical=ok_te, train_reason=why_tr, eval_reason=why_te,
                weights_identical=weights_equal, columns=len(ours_tr.columns),
                passes=bool(ok_tr and ok_te and weights_equal))
