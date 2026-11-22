#!/usr/bin/env python3
"""Saved Gefion loss curves, with restarts and unequal training budgets kept separate.

No GPU, model imports, fitting, smoothing or extrapolation. The old scans-only
runs were configured for 2000 epochs; the selected Boson comparisons used 150.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path('/dcai/projects/cu_0080/results/checkpoints')
ARMS = ('gaussian_kl', 'gaussian_mmd', 'boson_mmd')
LABELS = {'gaussian_kl': 'Gaussian / KL', 'gaussian_mmd': 'Gaussian / MMD',
          'boson_mmd': 'Boson / MMD'}
COLORS = {'gaussian_kl': '#1A8C91', 'gaussian_mmd': '#D38C36', 'boson_mmd': '#8161AF'}
COMPONENTS = {'loss': 'Total objective', 'chamfer': 'Reconstruction (Chamfer)',
              'binary_loss': 'Binary auxiliary loss (BCE)',
              'regression_loss': 'Regression auxiliary loss',
              'kl': 'KL regularizer', 'prior_match': 'MMD regularizer'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def fingerprint(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#E3E8EB', linewidth=.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Epoch')
    ax.tick_params(labelsize=9)


def save_figure(fig, stem, footer):
    fig.text(.5, .015, footer, ha='center', fontsize=8, color='#4E606B')
    fig.tight_layout(rect=(0, .075, 1, .93))
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(stem)+suffix, dpi=300, bbox_inches='tight', facecolor='white')
    pyplot().close(fig)


def option_map(command):
    """Read saved argv without importing the training parser or its dependencies."""
    if not isinstance(command, list) or len(command) < 2:
        raise ValueError('Saved train_command.json must be an argv list.')
    options, key = {}, None
    for token in map(str, command[2:]):
        if token.startswith('--'):
            key, equal, value = token.partition('=')
            if key in options:
                raise ValueError(f'Duplicate saved training option: {key}')
            options[key] = [value] if equal else []
        elif key is not None:
            options[key].append(token)
        else:
            raise ValueError('Unexpected positional argument in saved training command.')
    return options


def scalar(options, key, default=None):
    values = options.get(key)
    return values[0] if values else default


def run_metadata(run):
    metadata, inputs = {'run': run.name}, []
    for relative, label in [('training_summary.json', 'training_summary'),
                            ('study/status.json', 'study_status'),
                            ('study/train_command.json', 'command')]:
        path = run/relative
        if path.is_file():
            value = read_json(path)
            if label == 'training_summary':
                value = {key: item for key, item in value.items() if key != 'validation_ids'}
            elif label == 'study_status':
                value = {'status': value.get('status', 'unknown')}
            metadata[label] = value
            inputs.append(fingerprint(path))
    command = metadata.get('command')
    if command:
        options = option_map(command)
        metadata['configured_epochs'] = int(scalar(options, '--niter', 50))+int(scalar(options, '--niter_decay', 0))
        metadata['budget_source'] = 'saved training command'
        prior = scalar(options, '--latent_prior', 'gaussian')
        match = scalar(options, '--prior_match_mode', 'auto')
        match = ('mmd' if prior == 'boson' else 'kl') if match == 'auto' else match
        metadata['arm'] = f'{prior}_{match}'
    else:
        summary = metadata.get('training_summary', {})
        prior, match = summary.get('latent_prior', 'unknown'), summary.get('prior_match_mode', 'unknown')
        if match == 'auto':
            match = 'mmd' if prior == 'boson' else 'kl'
        metadata['arm'] = f'{prior}_{match}'
    if re.fullmatch(r'GEFION_SCANONLY_ANYPLAQ_[FM]_paper_v1_seed42', run.name):
        metadata.setdefault('configured_epochs', 2000)
        metadata.setdefault('budget_source', 'old submitted script default: 500 + 1500; command not saved')
    # Do not export full manifest: its members contain participant identifiers.
    manifest_path = run/'study'/'manifest.json'
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        metadata['manifest_summary'] = {key: manifest[key] for key in
            ('experiment', 'gender', 'fingerprint', 'reference') if key in manifest}
        if 'reference' in metadata['manifest_summary']:
            metadata['manifest_summary']['reference'] = {
                key: value for key, value in metadata['manifest_summary']['reference'].items()
                if key != 'members'}
        inputs.append(fingerprint(manifest_path))
    metadata['inputs'] = inputs
    return metadata


def epoch_history(train_path, validation_path):
    """One timestamp only; training is the mean of recorded minibatch losses."""
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path) if validation_path.is_file() else None
    omitted = []
    for frame, path in ((train, train_path), (validation, validation_path)):
        if frame is None:
            continue
        if frame.empty or not {'epoch', 'loss'} <= set(frame):
            raise ValueError(f'Missing epoch/loss values: {path}')
        for column in ['epoch', *[key for key in COMPONENTS if key in frame]]:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
        if (not np.isfinite(frame[['epoch', 'loss']].to_numpy(float)).all()
                or (frame.epoch < 1).any() or (frame.epoch % 1 != 0).any()
                or (np.diff(frame.epoch) < 0).any()):
            raise ValueError(f'Invalid or non-monotonic epoch/loss history: {path}')
        for column in COMPONENTS:
            if column in frame and not np.isfinite(frame[column].to_numpy(float)).all():
                if column != 'loss':
                    omitted.append(f'{path.name}: nonfinite {column}; component omitted')
                    frame.drop(columns=column, inplace=True)
    if 'iter' in train and train.duplicated(['epoch', 'iter']).any():
        raise ValueError(f'Duplicate epoch/minibatch rows: {train_path}')
    if validation is not None and (validation.epoch.duplicated().any()
            or not set(validation.epoch) <= set(train.epoch)):
        raise ValueError('Validation epochs are duplicated or do not belong to this timestamp.')
    keys = [key for key in COMPONENTS if key in train]
    result = train.groupby('epoch')[keys].mean().add_prefix('train_')
    result['recorded_minibatches'] = train.groupby('epoch').size()
    if validation is not None:
        keys = [key for key in COMPONENTS if key in validation]
        result = result.join(validation.set_index('epoch')[keys].add_prefix('validation_'))
    result.index = result.index.astype(int)
    return result, omitted


def plot_one(history, metadata, timestamp, stem):
    plt = pyplot()
    keys = [key for key in COMPONENTS if f'train_{key}' in history]
    columns = min(3, len(keys)); rows = int(np.ceil(len(keys)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.6*columns, 3.25*rows), squeeze=False)
    color = COLORS.get(metadata['arm'], '#1A8C91')
    for ax, key in zip(axes.flat, keys):
        ax.plot(history.index, history[f'train_{key}'], color=color, lw=1.2, label='Training')
        if f'validation_{key}' in history:
            ax.plot(history.index, history[f'validation_{key}'], color='#2C4050', ls='--', lw=1.2, label='Validation')
        ax.set_title(COMPONENTS[key], fontsize=11); style(ax); ax.legend(frameon=False, fontsize=8)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    budget = metadata.get('configured_epochs', 'unknown')
    fig.suptitle(f"{metadata['run']}\nHistory {timestamp} | recorded to epoch {history.index.max()} | configured budget {budget}", fontsize=11)
    save_figure(fig, stem, 'No smoothing or extrapolation. Training: minibatch means; validation: participant means.\n'
                'Different sampling modes; total objectives with KL and MMD are not the same quantity.')


def selected_names(reference_tag, boson_tag):
    for suite, experiment in [('weight', 'binary_age'), ('plaque', 'all_log')]:
        for gender in ('F', 'M'):
            names = {arm: f'GEFION_NEXT_{suite}_{experiment}{"" if arm == "gaussian_kl" else "_"+arm}_{gender}_{reference_tag if arm == "gaussian_kl" else boson_tag}_seed42'
                     for arm in ARMS}
            yield experiment, gender, names


def verify_matched_runs(root, names):
    """Check exact tasks, members and saved training settings, without exporting IDs."""
    original = root/names['gaussian_kl']
    base = read_json(original/'study'/'manifest.json')
    base_options = option_map(read_json(original/'study'/'train_command.json'))
    permitted = {'--name', '--checkpoints_dir', '--cond_tsv', '--val_cond_tsv', '--latent_prior',
                 '--prior_match_mode', '--mmd_buffer_size', '--mmd_prior_samples', '--deterministic',
                 '--boson_params', '--boson_sampler_path'}
    compare = lambda options: {key: value for key, value in options.items() if key not in permitted}
    if run_metadata(original)['arm'] != 'gaussian_kl':
        raise ValueError('The named historical reference is not Gaussian/KL.')
    for arm in ('gaussian_mmd', 'boson_mmd'):
        run = root/names[arm]
        manifest = read_json(run/'study'/'manifest.json')
        reference = manifest.get('reference', {})
        for key in ('experiment', 'gender', 'members'):
            if manifest.get(key) != base.get(key) or key not in base:
                raise ValueError(f'{run.name}: different or missing {key}')
        if reference.get('fingerprint') != base.get('fingerprint') or not base.get('fingerprint'):
            raise ValueError(f'{run.name}: reference fingerprint differs')
        for split in ('train', 'validation', 'test'):
            expected = reference.get('split_file_sha256', {}).get(split)
            if not expected or hashlib.sha256((original/'study'/f'{split}.tsv').read_bytes()).hexdigest() != expected:
                raise ValueError(f'{run.name}: original {split} data changed or hash unavailable')
        options = option_map(read_json(run/'study'/'train_command.json'))
        if compare(base_options) != compare(options) or run_metadata(run)['arm'] != arm:
            raise ValueError(f'{run.name}: training settings differ beyond prior controls')
    return {'status': 'matched', 'checks': 'Exact experiment, sex, split memberships, original split hashes and saved non-prior options.'}


def plot_comparison(histories, metadata, experiment, gender, out):
    plt = pyplot()
    colors = COLORS
    # Reconstruction/head losses share their definitions within these verified runs.
    head = 'binary_loss' if 'binary_age' in experiment else 'regression_loss'
    common = [key for key in ('chamfer', head)
              if all(f'train_{key}' in frame for frame in histories.values())]
    if common:
        fig, axes = plt.subplots(1, len(common), figsize=(6*len(common), 4), squeeze=False)
        for ax, key in zip(axes.flat, common):
            for arm, frame in histories.items():
                ax.plot(frame.index, frame[f'train_{key}'], color=colors[arm], lw=1.2, label=LABELS[arm]+' train')
                if f'validation_{key}' in frame:
                    ax.plot(frame.index, frame[f'validation_{key}'], color=colors[arm], ls='--', lw=1.2,
                            label=LABELS[arm]+' validation')
            ax.set_title(COMPONENTS[key], fontsize=12); style(ax)
        axes.flat[-1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Matched prior comparison | {experiment} | {'Women' if gender == 'F' else 'Men'}", fontsize=13)
        save_figure(fig, out/f'{experiment}_{gender}_matched_components',
                    'Same task, data splits and configured epoch budget; no smoothing.\n'
                    'Training: recorded minibatch means; validation: participant means.')
    # Never overlay KL and MMD objective/regularizer magnitudes as equivalent losses.
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7), squeeze=False)
    for column, arm in enumerate(ARMS):
        frame = histories[arm]
        regularizer = 'kl' if arm == 'gaussian_kl' else 'prior_match'
        for row, key in enumerate(('loss', regularizer)):
            ax = axes[row, column]
            for prefix, dash, label in [('train', '-', 'Training'), ('validation', '--', 'Validation')]:
                name = f'{prefix}_{key}'
                if name in frame:
                    ax.plot(frame.index, frame[name], color=colors[arm], ls=dash, lw=1.1, label=label)
            ax.set_title(f'{LABELS[arm]}: {COMPONENTS[key]}', fontsize=10)
            style(ax); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f'{experiment} | {gender} | objective and regularizer (separate axes)', fontsize=13)
    save_figure(fig, out/f'{experiment}_{gender}_objectives_separate',
                'KL and MMD have different definitions and scales; total objectives are not directly comparable.\n'
                'No smoothing, normalization or extrapolation. Assess shared reconstruction/task components above.')


def planned_long_comparisons(plan, histories, out):
    """Overlay new equal-budget fits using immutable clean-plan provenance."""
    result = []
    for gender in ('F', 'M'):
        entry = {'experiment': 'binary_age_long2000', 'gender': gender}
        tasks = {task.get('long_prior_arm'): task for task in plan.get('tasks', [])
                 if task.get('long_prior_arm') in ARMS and task.get('group') == gender}
        if set(tasks) != set(ARMS):
            result.append({**entry, 'status': 'not_overlaid', 'reason': 'Three planned long prior tasks are not available.'})
            continue
        entry['runs'] = {arm: task['run_name'] for arm, task in tasks.items()}
        try:
            cohort = next(row for row in plan['cohorts'] if row['target'] == 'peripheral_total' and row['group'] == gender)
            for path, expected in {**cohort['source_sha256'], **cohort['split_sha256']}.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Original/prepared split source changed: {path}')
            permitted = {'--name', '--latent_prior', '--prior_match_mode', '--mmd_buffer_size',
                         '--mmd_prior_samples', '--boson_params', '--boson_sampler_path'}
            settings, frames, timestamps = {}, {}, {}
            for arm, task in tasks.items():
                options = option_map(task['train_command'])
                if (int(scalar(options, '--niter', 0)) != 500 or int(scalar(options, '--niter_decay', 0)) != 1500
                        or '--deterministic' not in options or '--no-deterministic' in options):
                    raise ValueError('Long priors must share the deterministic 500+1500 epoch schedule.')
                prior = scalar(options, '--latent_prior', 'gaussian')
                match = scalar(options, '--prior_match_mode', 'auto')
                if f'{prior}_{match}' != arm:
                    raise ValueError('Planned latent prior/regularizer differs from its arm label.')
                if arm != 'gaussian_kl' and (scalar(options, '--mmd_buffer_size') != '64'
                                             or scalar(options, '--mmd_prior_samples') != '64'):
                    raise ValueError('MMD buffers/prior draws differ from the matched design.')
                settings[arm] = {key: value for key, value in options.items() if key not in permitted}
                payload = {key: task[key] for key in ('target', 'group', 'variant', 'train_command', 'evaluate_command')}
                payload['split_sha256'] = cohort['split_sha256']
                actual = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
                if actual != task['fingerprint']:
                    raise ValueError('Long task definition no longer matches its saved fingerprint.')
                marker = Path(task['run_dir'])/'clean_comparison_training_completed.json'
                status = Path(task['cohort_dir'])/(task['variant']+'_status.json')
                provenance = read_json(marker if marker.is_file() else status)
                if provenance.get('fingerprint') != task['fingerprint']:
                    raise ValueError('Executed task marker does not match this plan.')
                candidates = [(key, frame) for key, frame in histories.items() if key[0] == task['run_name']]
                if len(candidates) != 1:
                    raise ValueError('Need one unambiguous valid history per arm; restarts remain separate.')
                (name, timestamp), frame = candidates[0]
                frames[arm], timestamps[arm] = frame, timestamp
                if frame.index.max() > 2000:
                    raise ValueError('Recorded epoch exceeds the planned 2000-epoch budget.')
            if settings['gaussian_kl'] != settings['gaussian_mmd'] or settings['gaussian_kl'] != settings['boson_mmd']:
                raise ValueError('Long training settings differ beyond the planned prior controls.')
            plot_comparison(frames, {}, 'binary_age_long2000', gender, out)
            entry.update(status='matched', history_timestamps=timestamps,
                         checks='Exact prepared cohort hashes, immutable task fingerprints, executed task markers, identical non-prior training settings and 2000-epoch budgets.')
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            entry.update(status='not_overlaid', reason=str(exc))
        result.append(entry)
    return result


def discover(root, explicit, reference_tag, boson_tag):
    names = {f'GEFION_SCANONLY_ANYPLAQ_{gender}_paper_v1_seed42' for gender in ('F', 'M')}
    for _, _, selection in selected_names(reference_tag, boson_tag):
        names.update(selection.values())
    names.update(explicit)
    if root.is_dir():
        # Include nonstandard long/Boson histories without guessing their task.
        for run in sorted(root.iterdir()):
            if not run.is_dir() or not run.name.startswith('GEFION_'):
                continue
            if run.name.startswith(('GEFION_LONG_', 'GEFION_SCANONLY_')):
                names.add(run.name)
            elif (run/'metrics').is_dir():
                try:
                    meta = run_metadata(run)
                    if meta.get('arm', '').startswith('boson') or meta.get('configured_epochs', 0) > 150:
                        names.add(run.name)
                except (OSError, ValueError, TypeError, KeyError):
                    pass
    for name in names:
        if Path(name).name != name or name in ('.', '..') or '/' in name or '\\' in name:
            raise ValueError('--run must be a single directory name directly under --checkpoints-dir.')
    return sorted(names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints-dir', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=Path('requested_figures/gefion_loss_comparison'))
    parser.add_argument('--reference-tag', default='next_v1')
    parser.add_argument('--boson-tag', default='best_boson_v1')
    parser.add_argument('--run', action='append', default=[], help='Additional exact run directory name; repeatable.')
    parser.add_argument('--comparison-plan', type=Path, help='Clean workflow plan.json for new matched 2000-epoch prior runs.')
    parser.add_argument('--export-zip', type=Path, help='Optional ZIP of loss CSVs and sanitized provenance only (no participant data or weights).')
    args = parser.parse_args(argv)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    records, histories, metadata = [], {}, {}
    plan = read_json(args.comparison_plan) if args.comparison_plan else None
    plan_tasks = {task['run_name']: task for task in (plan or {}).get('tasks', []) if task.get('long_prior_arm') in ARMS}
    for name in discover(args.checkpoints_dir, [*args.run, *plan_tasks], args.reference_tag, args.boson_tag):
        run = args.checkpoints_dir/name
        paths = sorted((run/'metrics').glob('train_metrics_*.csv'))
        if not paths:
            records.append({'run': name, 'status': 'missing_history', 'reason': 'No train_metrics_*.csv'})
            print(f'[MISSING] {name}: no saved training history')
            continue
        try:
            meta = run_metadata(run); metadata[name] = meta
            if name in plan_tasks:
                task = plan_tasks[name]
                meta.update(arm=task['long_prior_arm'], configured_epochs=task['configured_epochs'],
                            budget_source='immutable clean comparison plan (executed marker checked before overlay)',
                            command=task['train_command'], task_fingerprint=task['fingerprint'])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            records.append({'run': name, 'status': 'invalid_metadata', 'reason': str(exc)})
            meta = {'run': name, 'arm': 'unknown'}; metadata[name] = meta
        for train in paths:
            timestamp = train.stem.removeprefix('train_metrics_')
            validation = train.with_name(f'val_metrics_{timestamp}.csv')
            record = {'run': name, 'timestamp': timestamp, 'arm': meta['arm'],
                      'configured_epochs': meta.get('configured_epochs'),
                      'budget_source': meta.get('budget_source', 'unknown')}
            try:
                frame, omitted = epoch_history(train, validation)
                histories[(name, timestamp)] = frame
                stem = out/f'{name}_{timestamp}'
                frame.to_csv(str(stem)+'_epochs.csv')
                plot_one(frame, meta, timestamp, stem)
                inputs = [fingerprint(path) for path in (train, validation) if path.is_file()]
                record.update(status='plotted' if validation.is_file() else 'plotted_training_only',
                              first_epoch=int(frame.index.min()), last_epoch=int(frame.index.max()),
                              recorded_epochs=len(frame), inputs=inputs, omitted_components=omitted)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                record.update(status='invalid_history', reason=str(exc))
            records.append(record)
            print(f"[{record['status'].upper()}] {name} / {timestamp}")
    comparisons = []
    for experiment, gender, names in selected_names(args.reference_tag, args.boson_tag):
        entry = {'experiment': experiment, 'gender': gender, 'runs': names}
        available = {arm: [(key, value) for key, value in histories.items() if key[0] == name]
                     for arm, name in names.items()}
        counts = {arm: len(values) for arm, values in available.items()}
        if any(count != 1 for count in counts.values()):
            entry.update(status='not_overlaid', reason='Need one unambiguous valid history per arm; restarts were plotted separately.', histories_per_arm=counts)
        else:
            try:
                entry.update(verify_matched_runs(args.checkpoints_dir, names))
                frames = {arm: values[0][1] for arm, values in available.items()}
                plot_comparison(frames, metadata, experiment, gender, out)
                entry['history_timestamps'] = {arm: values[0][0][1] for arm, values in available.items()}
            except (OSError, ValueError, TypeError, KeyError) as exc:
                entry.update(status='not_overlaid', reason=f'Matching could not be verified: {exc}')
        comparisons.append(entry)
        print(f"[{entry['status'].upper()}] matched {experiment} / {gender}")
    if plan:
        comparisons.extend(planned_long_comparisons(plan, histories, out))
    long_boson = [row for row in records if row.get('arm', '').startswith('boson')
                  and (row.get('configured_epochs') or 0) >= 2000]
    notes = {'runs': metadata, 'histories': records, 'matched_comparisons': comparisons,
             'long_boson_status': 'found_configured_2000_or_more' if long_boson else 'no_verified_2000_epoch_boson_run_found',
             'known_budgets': 'Old scan-only default: 2000 epochs. Selected prior study default: 150 epochs; it is not a long-run Boson comparison.',
             'aggregation': 'Training: arithmetic mean of recorded minibatches each epoch; validation: stored participant means.',
             'restarts': 'Each timestamp is independent. No latest-file guessing, merging, smoothing, padding or extrapolation.',
             'interpretation': 'KL and MMD total objectives/regularizers are different. Shared reconstruction/task losses are compared only after exact task/split/settings checks.',
             'zip_scope': 'Only known numeric loss columns and sanitized provenance; no participant identifiers, split TSVs, predictions or weights.'}
    if args.comparison_plan:
        notes['comparison_plan_source'] = fingerprint(args.comparison_plan)
    write_json(out/'loss_comparison_audit.json', notes)
    inventory = pd.DataFrame([{key: value for key, value in row.items() if key not in ('inputs', 'omitted_components')}
                              for row in records])
    inventory.to_csv(out/'loss_inventory.csv', index=False)
    if args.export_zip:
        # Write re-parsed numeric-only histories rather than raw arbitrary CSV columns.
        args.export_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.export_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('loss_comparison_audit.json', json.dumps(notes, indent=2, allow_nan=False))
            archive.writestr('loss_inventory.csv', inventory.to_csv(index=False))
            for (name, timestamp), frame in histories.items():
                archive.writestr(f'{name}/{timestamp}_epochs.csv', frame.to_csv())
        print(f'[BUNDLE] {args.export_zip.resolve()}')
    if not long_boson:
        print('[NOTE] No verified 2000-epoch Boson run found. Existing selected prior runs are kept separate from old long runs.')
    print('[OUT]', out.resolve())
    return notes


if __name__ == '__main__':
    main()
