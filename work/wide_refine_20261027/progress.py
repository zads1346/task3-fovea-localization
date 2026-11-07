"""Lightweight local progress snapshot; no model imports and no GPU access."""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import json
import math

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wide_refine_20261027'
STAGES = ('wide_select', 'wide_dev', 'wide_full')
SEEDS = (2026, 2027)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, allow_empty=False):
    if allow_empty and (value is None or value == ''):
        return None
    result = float(value)
    check(math.isfinite(result), f'Nonfinite {label}')
    return result


def snapshot():
    completed, folds, errors = [], [], []
    for seed in SEEDS:
        for fold in range(5):
            folder = OUT / f'split_{seed}' / f'fold_{fold}'
            for stage in STAGES:
                stage_dir = folder / stage
                marker = stage_dir / 'stage_complete.json'
                if not marker.is_file():
                    continue
                relative = stage_dir.relative_to(OUT).as_posix()
                try:
                    record = read_json(marker)
                    check(bool(record.get('signature')), f'{relative}: missing stage signature')
                    summary = read_json(stage_dir / 'training_summary.json')
                    history = read_csv(stage_dir / 'history.csv')
                    check(bool(history), f'{relative}: empty completed history')
                    epochs = [int(row['epoch']) for row in history]
                    check(epochs == list(range(1, len(history) + 1)), f'{relative}: noncontiguous epochs')
                    updates = int(summary['successful_optimizer_steps'])
                    check(updates > 0, f'{relative}: no successful optimizer updates')
                    row_updates = [int(row['successful_optimizer_steps']) for row in history]
                    check(all(value >= 0 for value in row_updates) and sum(row_updates) == updates,
                          f'{relative}: inconsistent optimizer update count')
                    for row in history:
                        number(row['train_loss'], relative + ' training loss')
                        number(row['val_rmse'], relative + ' validation RMSE', allow_empty=True)
                        number(row['seconds'], relative + ' epoch duration')
                    fixed = bool(summary['fixed_epochs'])
                    check(fixed == (stage != 'wide_select'), f'{relative}: wrong selection/refit mode')
                    if fixed:
                        check(len(history) == int(summary['requested_training_epochs']),
                              f'{relative}: incomplete fixed-epoch refit')
                    else:
                        check(all(row['val_rmse'] != '' for row in history),
                              f'{relative}: missing early-stop validation score')
                    completed.append({'split_seed': seed, 'fold': fold, 'fold_display': fold + 1,
                        'stage': stage, 'path': relative, 'training_images': int(summary['training_images']),
                        'completed_epochs': len(history), 'selected_epoch': int(summary['best_epoch']),
                        'best_rmse': number(summary['best_rmse'], relative + ' best RMSE', allow_empty=True),
                        'successful_optimizer_steps': updates,
                        'last_train_loss': number(history[-1]['train_loss'], relative + ' last loss'),
                        'last_validation_rmse': number(history[-1]['val_rmse'], relative + ' last RMSE', allow_empty=True),
                        'marker_modified_ns': marker.stat().st_mtime_ns})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append({'path': relative, 'error': str(exc)})
            marker = folder / 'complete.json'
            if marker.is_file():
                try:
                    record = read_json(marker)
                    check(record['status'] == 'COMPLETE', f'{folder}: invalid fold status')
                    check(record['split_seed'] == seed and record['fold'] == fold, 'Fold marker identity mismatch')
                    decision = read_json(folder / 'selection.json')
                    rows = read_csv(folder / 'metrics.csv')
                    values = {row['method']: number(row['rmse'], 'fold metric RMSE') for row in rows}
                    folds.append({'split_seed': seed, 'fold': fold, 'fold_display': fold + 1,
                                  'accepted': bool(decision['accepted']),
                                  'selected_epoch': int(decision['selected_epoch']),
                                  'selection_gain_px': number(decision['gain_px'], 'selection gain'),
                                  'oof_rmse': values})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append({'path': folder.relative_to(OUT).as_posix(), 'error': str(exc)})
    latest = sorted(completed, key=lambda row: row['marker_modified_ns'], reverse=True)[:3]
    for item in completed:
        item.pop('marker_modified_ns', None)
    execution = read_json(OUT / 'execution_report.json') if (OUT / 'execution_report.json').is_file() else None
    status = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'completed_stages': len(completed), 'total_stages': 30,
              'completed_folds': len(folds), 'total_folds': 10,
              'completed_stage_history_checks': 'PASS' if not errors else 'FAIL',
              'latest_completed_stages': latest, 'folds': folds,
              'execution_status': execution.get('status') if execution else 'RUNNING_OR_NOT_STARTED',
              'errors': errors,
              'scope': 'Atomic completion markers and completed training histories only; active epochs are shown in training.log.'}
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brief', action='store_true', help='Print a compact progress line and latest completed stage.')
    args = parser.parse_args()
    status = snapshot()
    if OUT.is_dir():
        temporary = OUT / 'live_progress.partial.json'
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(OUT / 'live_progress.json')
    if args.brief:
        print(f"STAGES {status['completed_stages']}/30 | FOLDS {status['completed_folds']}/10 | "
              f"CHECKS {status['completed_stage_history_checks']} | {status['execution_status']}")
        if status['latest_completed_stages']:
            last = status['latest_completed_stages'][0]
            print(f"Latest: {last['path']} | epochs={last['completed_epochs']} | "
                  f"successful_updates={last['successful_optimizer_steps']} | loss={last['last_train_loss']:.6f}")
        for error in status['errors']:
            print('ERROR:', error['path'], error['error'])
    else:
        print(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False))
    if status['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
