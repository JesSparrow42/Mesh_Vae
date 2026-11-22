#!/usr/bin/env python3
"""Add six new, matched 2000-epoch prior comparisons to a clean Gefion plan.

The existing 150-epoch Boson fits are not extended or relabelled as long runs.
This component uses the supplied standalone comparison module, never a project
import. The same runner trains and evaluates these ordinary plan tasks.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ARMS = ('gaussian_kl', 'gaussian_mmd', 'boson_mmd')


def append_long_tasks(plan, comparison):
    """Return a plan with F/M x three priors; existing short tasks are unchanged."""
    plan = copy.deepcopy(plan)
    existing = {(task['target'], task['group'], task['variant']): task for task in plan['tasks']}
    additions = []
    for gender in ('F', 'M'):
        key = ('peripheral_total', gender, 'scans_age')
        if key not in existing:
            raise ValueError(f'Long comparison requires a prepared peripheral_total/{gender}/scans_age task.')
        original = existing[key]
        cohort = next(row for row in plan['cohorts'] if row['target'] == 'peripheral_total' and row['group'] == gender)
        if cohort['status'] == 'unavailable':
            raise ValueError(f'Long comparison cohort unavailable: {gender}')
        comparison.verify_cohort(cohort)
        for arm in ARMS:
            variant = 'long_' + arm
            name = f"GEFION_LONG_peripheral_total_scans_age_{arm}_{gender}_{plan['run_tag']}_seed42"
            options = comparison.command_options(original['train_command'])
            for key, setting in {'name': name, 'niter': 500, 'niter_decay': 1500,
                                 'latent_prior': 'boson' if arm == 'boson_mmd' else 'gaussian',
                                 'prior_match_mode': 'kl' if arm == 'gaussian_kl' else 'mmd',
                                 'mmd_buffer_size': 64, 'mmd_prior_samples': 64}.items():
                comparison.set_value(options, key, setting)
            comparison.set_flag(options, 'deterministic', True)
            options.pop('--boson_params', None)
            options.pop('--boson_sampler_path', None)
            if arm == 'boson_mmd':
                comparison.set_value(options, 'boson_params', '{}')
                root = Path(original['train_command'][1]).parent
                comparison.set_value(options, 'boson_sampler_path', root/'third_party'/'nonlinear_photonic_sampler')
            command = list(original['train_command'][:2])
            for option, values in options.items():
                command.extend([option, *values])
            run_dir = str(Path(plan['checkpoints_dir'])/name)
            task = {'task_index': len(plan['tasks'])+len(additions), 'target': 'peripheral_total',
                    'endpoint': original['endpoint'], 'group': gender, 'variant': variant,
                    'run_name': name, 'run_dir': run_dir, 'cohort_dir': original['cohort_dir'],
                    'train_command': command,
                    'evaluate_command': comparison.evaluation_command(command, Path(cohort['splits_dir'])/'test.tsv',
                                                                         root=Path(command[1]).parent),
                    'predictions_path': str(Path(original['cohort_dir'])/'predictions'/f'{variant}.csv'),
                    'status': 'pending', 'seed': 42, 'configured_epochs': 2000,
                    'training_inputs': ['scans', 'age'], 'training_labels': [original['endpoint']],
                    'reuse_audit': None, 'long_prior_arm': arm,
                    'comparison': 'New matched long experiment; no historical short/long checkpoint is reused.'}
            task['fingerprint'] = comparison.fingerprint({key: task[key] for key in
                ('target', 'group', 'variant', 'train_command', 'evaluate_command')}
                | {'split_sha256': cohort['split_sha256']})
            previous = existing.get(('peripheral_total', gender, variant))
            if previous:
                if previous['fingerprint'] != task['fingerprint']:
                    raise ValueError('Existing long-task settings changed; choose a fresh run tag/output directory.')
                continue
            additions.append(task)
    plan['tasks'].extend(additions)
    plan['long_comparison'] = {'configured_epochs': 2000, 'niter': 500, 'niter_decay': 1500,
        'groups': ['F', 'M'], 'arms': list(ARMS), 'endpoint': 'presence_peripheral_total',
        'inputs': ['scans', 'age'], 'seed': 42,
        'scope': 'Six new fits with exact shared F/M train/validation/test splits and equal training budgets.',
        'interpretation': 'Gaussian-MMD vs Boson-MMD compares prior distributions. Gaussian-KL is an alternate regularizer; total KL/MMD objectives differ.'}
    comparison.write_json(Path(plan['output_dir'])/'plan.json', plan)
    return plan


def main(argv=None, comparison=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(argv)
    if comparison is None:
        raise ValueError('Use the single-file clean Gefion workflow with --include-long; this is its embedded component.')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    return append_long_tasks(plan, comparison)


if __name__ == '__main__':
    main()
