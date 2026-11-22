#!/usr/bin/env python3
"""Matched classification with plaque-volume labels used only during training.

The existing fixed-width plaque head has four outputs. All four learn the SAME
standardized log1p(total peripheral volume), averaged by the existing SmoothL1
loss. The control keeps the identical head and random initialization but gives
this auxiliary loss weight zero. No core model edits or extra packages needed.
"""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess

TARGET = 'peripheral_total_volume_training'
ENDPOINT = 'presence_peripheral_total'
RAW_VOLUME = 'volume_peripheral_total'
AUX_COLUMN = 'volume_aux_target'
VARIANTS = ('volume_control', 'volume_aux')
WEIGHT = 0.25


def prepare_frames(frames, group, comparison):
    """Restrict only TRAIN/VALIDATION; fit target normalization using TRAIN."""
    import numpy as np
    import pandas as pd
    prepared, exclusions = {}, {}
    for split in comparison.SPLITS:
        frame = frames[split].copy()
        if RAW_VOLUME not in frame:
            raise ValueError(f'Missing measured total plaque volume in {split}: {RAW_VOLUME}')
        volume = pd.to_numeric(frame[RAW_VOLUME], errors='coerce').to_numpy(float)
        observed = np.isfinite(volume) & (volume >= 0)
        # A negative recorded volume is invalid, never silently a zero plaque.
        if (np.isfinite(volume) & (volume < 0)).any():
            raise ValueError(f'Negative measured plaque volume in {split}')
        if not np.array_equal((volume[observed] > 0).astype(int), frame.loc[observed, ENDPOINT].to_numpy(int)):
            raise ValueError(f'Presence label disagrees with complete measured volume in {split}')
        exclusions[split] = int((~observed).sum()) if split != 'test' else 0
        if split != 'test':
            frame = frame.loc[observed].copy()
        prepared[split] = comparison.validate_frame(frame, ENDPOINT, group)
    comparison.validate_disjoint(prepared)
    transformed = np.log1p(prepared['train'][RAW_VOLUME].to_numpy(float))
    mean, scale = float(transformed.mean()), float(transformed.std(ddof=0))
    if not np.isfinite([mean, scale]).all():
        raise ValueError('Nonfinite training-only plaque volume normalization')
    if scale < 1e-6:
        scale = 1.0
    for split, frame in prepared.items():
        # Held-out measured volumes never enter the auxiliary labels or inputs.
        frame[AUX_COLUMN] = 0.0 if split == 'test' else (np.log1p(frame[RAW_VOLUME].to_numpy(float))-mean)/scale
    transform = {'column': RAW_VOLUME, 'transform': 'log1p', 'mean': mean, 'scale': scale,
                 'fit_partition': 'train', 'fit_n': len(prepared['train']),
                 'training_validation_exclusions': exclusions,
                 'heldout_auxiliary_label': 'constant zero; measured volume is not required to classify',
                 'head': 'four outputs learning the same total-volume target; averaged SmoothL1',
                 'normalization_inside_loader': 'none; supplied values already use training-only statistics'}
    return prepared, transform


def auxiliary_command(original, name, paths, weight, comparison):
    options = comparison.command_options(original)
    for key, value in {'name': name, 'cond_tsv': paths['train'], 'val_cond_tsv': paths['validation'],
                       'aux_plaque_norm': 'none', 'aux_plaque_weight': weight}.items():
        comparison.set_value(options, key, value)
    options['--aux_plaque_cols'] = [AUX_COLUMN]*4
    comparison.set_flag(options, 'aux_predict_plaque', True)
    comparison.set_flag(options, 'deterministic', True)
    for key in ('cond_plaque', 'cond_athero', 'aux_predict_athero', 'aux_predict_regression', 'use_cond_prior'):
        comparison.set_flag(options, key, False)
    command = list(original[:2])
    for key, values in options.items():
        command.extend([key, *values])
    return command


def auxiliary_evaluation(command, test_path, comparison):
    result = comparison.evaluation_command(command, test_path, root=Path(command[1]).parent)
    # TestOptions does NOT restore auxiliary plaque head settings from metadata.
    # Keep the head for checkpoint compatibility; suppress irrelevant dummy loss.
    result.extend(['--aux_predict_plaque', '--aux_plaque_cols', *([AUX_COLUMN]*4),
                   '--aux_plaque_norm', 'none', '--aux_plaque_weight', '0', '--deterministic'])
    return result


def append_volume_tasks(plan, comparison):
    """Add up to six fits without changing the main figure's training cohorts."""
    import pandas as pd
    plan = copy.deepcopy(plan)
    output = Path(plan['output_dir'])
    existing = {(task['target'], task['group'], task['variant']): task for task in plan['tasks']}
    for group in ('F', 'M', 'combined'):
        original = existing.get(('peripheral_total', group, 'scans_age'))
        if original is None:
            continue
        source = next(c for c in plan['cohorts'] if c['target'] == 'peripheral_total' and c['group'] == group)
        frames = comparison.verify_cohort(source)
        frames, transform = prepare_frames(frames, group, comparison)
        base = output/TARGET/group
        cohort = {'target': TARGET, 'endpoint': ENDPOINT, 'group': group, 'status': 'prepared',
                  'experiment': 'training-only plaque-volume supervision',
                  'splits_dir': str(base/'splits'), 'predictions_dir': str(base/'predictions'),
                  'endpoint_definition': source['endpoint_definition'],
                  'reference_runs': source['reference_runs'], 'source_sha256': source['source_sha256'],
                  'members': {s: list(f.participant_id) for s, f in frames.items()},
                  'counts': {s: len(f) for s, f in frames.items()}, 'sex_predictor': source['sex_predictor'],
                  'prediction_paths': {v: str(base/'predictions'/f'{v}.csv') for v in VARIANTS},
                  'volume_transform': transform}
        paths = {'endpoint': ENDPOINT}
        for split, frame in frames.items():
            path = base/'splits'/f'{split}.tsv'
            comparison.immutable_text(path, frame.to_csv(sep='\t', index=False))
            paths[split] = str(path)
        cohort['split_sha256'] = {paths[s]: comparison.sha256(paths[s]) for s in comparison.SPLITS}
        transform_path = base/'volume_transform.json'
        comparison.write_json(transform_path, transform)
        cohort['source_sha256'] = {**cohort['source_sha256'], str(transform_path): comparison.sha256(transform_path)}
        prior_cohort = next((c for c in plan['cohorts'] if c['target'] == TARGET and c['group'] == group), None)
        if prior_cohort is None:
            plan['cohorts'].append(cohort)
        elif comparison.fingerprint(prior_cohort) != comparison.fingerprint(cohort):
            raise ValueError('Volume supervision cohort changed; use a fresh run tag/output directory')
        for variant, weight in zip(VARIANTS, (0., WEIGHT)):
            name = f"GEFION_CLEAN_{TARGET}_{variant}_{group}_{plan['run_tag']}_seed42"
            command = auxiliary_command(original['train_command'], name, paths, weight, comparison)
            task = {'task_index': len(plan['tasks']), 'target': TARGET, 'endpoint': ENDPOINT,
                    'group': group, 'variant': variant, 'run_name': name,
                    'run_dir': str(Path(plan['checkpoints_dir'])/name), 'cohort_dir': str(base),
                    'train_command': command, 'evaluate_command': auxiliary_evaluation(command, paths['test'], comparison),
                    'predictions_path': cohort['prediction_paths'][variant], 'status': 'pending', 'seed': 42,
                    'configured_epochs': original['configured_epochs'], 'training_inputs': ['scans', 'age'],
                    'training_labels': [ENDPOINT]+([AUX_COLUMN] if weight else []),
                    'auxiliary_loss_weight': weight, 'auxiliary_head_outputs': 4, 'reuse_audit': None,
                    'comparison': 'Identical full-volume-observed training/validation participants, architecture and initialization; original presence-known test retained.',
                    'inference_check_required': True}
            task['fingerprint'] = comparison.fingerprint({k: task[k] for k in
                ('target', 'group', 'variant', 'train_command', 'evaluate_command')} | {'split_sha256': cohort['split_sha256']})
            previous = existing.get((TARGET, group, variant))
            if previous is not None:
                if previous['fingerprint'] != task['fingerprint']:
                    raise ValueError('Existing volume task settings differ; use fresh run tag/output directory')
            else:
                plan['tasks'].append(task)
    plan['volume_supervision'] = {'endpoint': ENDPOINT, 'variants': list(VARIANTS),
        'target_key': TARGET, 'auxiliary_loss': 'averaged SmoothL1 of standardized log1p total peripheral volume',
        'auxiliary_weight': WEIGHT, 'test_inputs': ['scans', 'age'], 'raw_volume_is_predictor': False,
        'interpretation': 'A matched training-supervision experiment; no measured plaque volume is supplied as an input at inference.'}
    comparison.write_json(output/'plan.json', plan)
    return plan


def check_inference_invariance(task, comparison):
    """GPU follow-up: changing held-out auxiliary labels must not change scores."""
    import numpy as np
    import pandas as pd
    if task.get('target') != TARGET:
        return
    base = Path(task['cohort_dir'])
    marker = base/(task['variant']+'_inference_check.json')
    if marker.exists():
        saved = comparison.read_json(marker)
        if saved.get('fingerprint') == task['fingerprint'] and saved.get('status') == 'passed':
            return
    test = pd.read_csv(base/'splits'/'test.tsv', sep='\t', dtype={'participant_id': str})
    original = comparison.exact_predictions(task['predictions_path'], test, ENDPOINT)
    altered = test.copy()
    altered[AUX_COLUMN] = np.linspace(-3., 3., len(altered))
    alternate_path = base/'splits'/'test_auxiliary_invariance.tsv'
    comparison.immutable_text(alternate_path, altered.to_csv(sep='\t', index=False))
    command = list(task['evaluate_command'])
    command[command.index('--cond_tsv')+1] = str(alternate_path)
    subprocess.run(command, check=True)
    alternate = comparison.diagnosis_predictions(Path(task['run_dir']), test, ENDPOINT)
    difference = np.abs(original.probability.to_numpy()-alternate.probability.to_numpy())
    if not np.allclose(original.probability, alternate.probability, rtol=1e-6, atol=1e-7):
        raise ValueError('Changing held-out auxiliary labels changed binary predictions; stop the comparison')
    comparison.write_json(marker, {'status': 'passed', 'fingerprint': task['fingerprint'],
        'test_n': len(test), 'max_absolute_probability_difference': float(difference.max()),
        'check': 'constant-zero auxiliary labels versus varied dummy labels; measured volume is not required'})

