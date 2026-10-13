"""Five predeclared candidates, frozen development folds; no final model selection."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline import run
from preprocess import write_json, sha256_file

CANDIDATES = {
    'lr_c01': ('lr', {'C': 0.1}),
    'lr_c10': ('lr', {'C': 10.0}),
    'rf_bal_mf01_leaf1': ('rf', {'max_features': 0.1, 'min_samples_leaf': 1}),
    'rf_bal_mf02_leaf1': ('rf', {'max_features': 0.2, 'min_samples_leaf': 1}),
    'rf_bal_mf02_leaf2': ('rf', {'max_features': 0.2, 'min_samples_leaf': 2}),
}

def validate_candidate(key: str, config: dict) -> str:
    if key not in CANDIDATES:
        raise ValueError(f'Not in the predeclared search space: {key}')
    model, changed = CANDIDATES[key]
    reference = 'baseline.json' if model == 'lr' else 'class_weights.json'
    base = json.loads((ROOT/'configs'/reference).read_text(encoding='utf-8'))
    expected = dict(base['models'][model], **changed)
    if config['models'] != {model: expected} or config.get('sampling') or config.get('sample_weight'):
        raise ValueError('Candidate differs from its declared fixed comparison.')
    if config['seed'] != 42 or config['metric'] != 'macro_f1' or config['name'] != 'tuning_v1_' + key:
        raise ValueError('Seed, metric or experiment identity mismatch.')
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/tuning_v1')
    parser.add_argument('--only', nargs='+', choices=list(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if len(args.only) != len(set(args.only)):
        parser.error('Duplicate candidates are not allowed.')
    configs = {}
    for key in args.only:
        config_path = ROOT/'configs/tuning'/f'{key}.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        configs[key] = (config_path, validate_candidate(key, config))
        path = args.output/key
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f'Result already exists: {path}. No automatic rerun or overwrite.')
    for key, (config_path, model) in configs.items():
        print(f'BEGIN {key}', flush=True)
        run(args.data, args.output/key, config_path, models=[model], jobs=args.jobs)
        write_json(args.output/key/'tuning_provenance.json', {
            'source_repository': 'Peter-jackson12/earthquake',
            'verified_modeling_base_commit': '2d1bba40a778b48fef8a41ef36c32802aefae794',
            'scope_commit': '607d5dc992ef442517cbbb727464dc5e57e51713',
            'runner_sha256': sha256_file(__file__),
            'config_sha256': sha256_file(config_path),
            'holdout_scored': False,
            'selection_scope': 'development CV only; not an unbiased selected-model test estimate'
        })
        print(f'END {key}', flush=True)

if __name__ == '__main__':
    main()
