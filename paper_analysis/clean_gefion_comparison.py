#!/usr/bin/env python3
"""Prepare and train matched age / scan / scan+age comparisons on Gefion.

This is a standalone upload: only numpy/pandas are imported, never a project
helper. GPU training/diagnosis is delegated to the existing repository entry
points. Saved NEXT study splits define the endpoint and participants exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = Path('/dcai/projects/cu_0080/results/checkpoints')
TERRITORIES = ('carotid_right', 'carotid_left', 'carotid_total', 'femoral_right',
               'femoral_left', 'femoral_total', 'coronary_volume')
SPLITS = ('train', 'validation', 'test')
VARIANTS = ('scans_only', 'scans_age')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_id(value):
    value = str(value).strip().strip('"').strip("'")
    if not value or value.lower() in ('nan', 'none', '<na>'):
        raise ValueError('Missing participant ID')
    match = re.search(r'(GLODET\d+)', value, flags=re.I)
    return match.group(1).upper() if match else value


def command_options(command):
    """Preserve nargs options and reject ambiguous duplicate options."""
    if not isinstance(command, list) or len(command) < 2 or not all(isinstance(x, str) for x in command):
        raise ValueError('train_command.json must contain a command argument list')
    result, current = {}, None
    for token in command[2:]:
        if token.startswith('--'):
            if '=' in token:
                key, value = token.split('=', 1)
                values = [value]
            else:
                key, values = token, []
            if key in result:
                raise ValueError(f'Duplicate command option {key}')
            result[key], current = values, key
        elif current is None:
            raise ValueError(f'Unexpected positional command value: {token}')
        else:
            result[current].append(token)
    for key in result:
        if key.startswith('--no-') and '--' + key[5:] in result:
            raise ValueError(f'Conflicting boolean options: {key}')
    return result


def value(options, key, default=None):
    values = options.get('--' + key)
    if values is None:
        return default
    if len(values) != 1:
        raise ValueError(f'Expected one value for --{key}')
    return values[0]


def flag(options, key, default=False):
    if '--no-' + key in options:
        if options['--no-' + key]:
            raise ValueError(f'Boolean --no-{key} unexpectedly has values')
        return False
    if '--' + key in options:
        if options['--' + key]:
            raise ValueError(f'Boolean --{key} unexpectedly has values')
        return True
    return default


def set_value(options, key, setting):
    options.pop('--no-' + key, None)
    options['--' + key] = [str(setting)]


def set_flag(options, key, setting):
    options.pop('--' + key, None)
    options.pop('--no-' + key, None)
    options[('--' if setting else '--no-') + key] = []


def reference_name(target, group, tag):
    middle = 'weight_binary_age' if target == 'peripheral_total' else f'territory_classification_{target}_presence'
    return f'GEFION_NEXT_{middle}_{group}_{tag}_seed42'


def validate_frame(frame, target, group=None):
    import numpy as np
    required = ['participant_id', 'study_age', 'study_gender', target]
    if any(x not in frame for x in required):
        raise ValueError(f'Missing split columns: {sorted(set(required) - set(frame))}')
    result = frame.copy()
    result['participant_id'] = result.participant_id.map(normalized_id)
    if result.empty or result.participant_id.duplicated().any():
        raise ValueError('Empty split or duplicate normalized participant IDs')
    if not np.isfinite(result[['study_age', target]].to_numpy(float)).all():
        raise ValueError('Nonfinite age or endpoint values')
    if (result.study_age.to_numpy(float) < 18).any():
        raise ValueError('A participant is younger than the existing loader minimum (18)')
    if not np.isin(result[target].to_numpy(float), [0, 1]).all():
        raise ValueError('Endpoint must be binary 0/1')
    if result[target].nunique() != 2:
        raise ValueError('Both endpoint classes are required in each split')
    if not result.study_gender.isin(['F', 'M']).all():
        raise ValueError('study_gender must preserve actual F/M values')
    if group in ('F', 'M') and not (result.study_gender == group).all():
        raise ValueError('Saved split contains another sex')
    return result


def validate_disjoint(frames):
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1:]:
            if set(frames[first].participant_id) & set(frames[second].participant_id):
                raise ValueError(f'Participant overlap between {first} and {second}')


def load_reference(checkpoints, target, group, tag):
    import pandas as pd
    name = reference_name(target, group, tag)
    run, endpoint = Path(checkpoints) / name, 'presence_' + target
    study = run / 'study'
    required = [study / 'manifest.json', study / 'status.json', study / 'train_command.json',
                *[study / f'{split}.tsv' for split in SPLITS]]
    for path in required:
        if not path.is_file():
            raise ValueError(f'Missing reference file: {path}')
    manifest = read_json(study / 'manifest.json')
    if manifest.get('gender') != group or manifest.get('experiment', {}).get('territory') != target:
        raise ValueError(f'Reference sex/endpoint mismatch: {run}')
    if manifest.get('experiment', {}).get('kind') != 'binary':
        raise ValueError(f'Reference is not binary classification: {run}')
    if manifest.get('config', {}).get('run_tag') != tag:
        raise ValueError(f'Reference run tag differs: {run}')
    if int(manifest.get('config', {}).get('seed', -1)) != 42:
        raise ValueError('The original reference split must use seed 42')
    command = read_json(study / 'train_command.json')
    options = command_options(command)
    if value(options, 'aux_binary_col') != endpoint or not flag(options, 'aux_predict_binary'):
        raise ValueError(f'Saved command uses another endpoint: {run}')
    if value(options, 'only_gender') != group:
        raise ValueError(f'Saved command sex filter differs: {run}')
    if value(options, 'feature_source', 'scan') != 'scan':
        raise ValueError('Reference must use scans')
    if options.get('--clinical_feature_cols'):
        raise ValueError('Reference unexpectedly uses clinical feature columns')
    if int(value(options, 'seed', 42)) != 42:
        raise ValueError('Reference command and manifest seeds differ')
    for option in ('niter', 'niter_decay', 'batch_size'):
        if int(value(options, option, 0)) != int(manifest['config'][option]):
            raise ValueError(f'Saved command/config disagree on {option}')
    if int(value(options, 'niter', 0)) + int(value(options, 'niter_decay', 0)) != 150:
        raise ValueError('Expected the 150-epoch NEXT comparison budget')
    frames = {}
    for split in SPLITS:
        path = study / f'{split}.tsv'
        frames[split] = validate_frame(pd.read_csv(path, sep='\t', dtype={'participant_id': str}), endpoint, group)
        members = [normalized_id(x) for x in manifest.get('members', {}).get(split, [])]
        if list(frames[split].participant_id) != members:
            raise ValueError(f'{split} participant order differs from reference manifest')
    validate_disjoint(frames)
    return {'name': name, 'run': run, 'study': study, 'frames': frames, 'command': command,
            'options': options, 'manifest': manifest, 'status': read_json(study / 'status.json'),
            'source_sha256': {str(path): sha256(path) for path in required}}


def same_settings(first, second):
    """Joint fits require compatible source settings, including the training data."""
    ignored = {'--name', '--cond_tsv', '--val_cond_tsv', '--only_gender'}
    a = {key: val for key, val in first['options'].items() if key not in ignored}
    b = {key: val for key, val in second['options'].items() if key not in ignored}
    if a != b:
        differences = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
        raise ValueError('Sex-specific reference commands differ beyond sex/split paths: ' + ', '.join(differences))


def combine_frames(female, male, endpoint):
    import pandas as pd
    all_f = set().union(*(set(female[k].participant_id) for k in SPLITS))
    all_m = set().union(*(set(male[k].participant_id) for k in SPLITS))
    if all_f & all_m:
        raise ValueError('The F/M reference partitions overlap across sexes')
    frames = {split: validate_frame(pd.concat([female[split], male[split]], ignore_index=True), endpoint)
              for split in SPLITS}
    validate_disjoint(frames)
    return frames


def age_predictions(train, test, endpoint):
    """Training-only standardized ridge logistic; same fixed penalty as NEXT."""
    import numpy as np
    import pandas as pd
    x = train[['study_age']].to_numpy(float)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    design = np.c_[np.ones(len(x)), (x - mean) / scale]
    test_design = np.c_[np.ones(len(test)), (test[['study_age']].to_numpy(float) - mean) / scale]
    target = train[endpoint].to_numpy(float)
    penalty = np.diag([0., 1.])
    beta = np.zeros(2)
    for iteration in range(100):
        probability = 1 / (1 + np.exp(-np.clip(design @ beta, -35, 35)))
        weight = np.maximum(probability * (1 - probability), 1e-8)
        step = np.linalg.solve(design.T @ (weight[:, None] * design) + penalty,
                               design.T @ (probability - target) + penalty @ beta)
        beta -= step
        if abs(step).max() < 1e-8:
            break
    else:
        raise ValueError('Age-only ridge logistic did not converge in 100 iterations')
    prediction = 1 / (1 + np.exp(-np.clip(test_design @ beta, -35, 35)))
    output = pd.DataFrame({'participant_id': test.participant_id, 'target': test[endpoint], 'probability': prediction})
    return output, {'model': 'ridge logistic', 'features': ['study_age'], 'ridge': 1.,
                    'intercept_penalized': False, 'training_age_mean': float(mean[0]),
                    'training_age_scale': float(scale[0]), 'coefficients': beta.tolist(),
                    'iterations': iteration + 1, 'fit_partition': 'train'}


def exact_predictions(path, test, endpoint, score_column='probability'):
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(path, dtype={'participant_id': str})
    required = ['participant_id', 'target', score_column]
    if any(x not in frame for x in required):
        raise ValueError(f'Incomplete prediction columns in {path}')
    frame['participant_id'] = frame.participant_id.map(normalized_id)
    if frame.participant_id.duplicated().any() or set(frame.participant_id) != set(test.participant_id):
        raise ValueError(f'Prediction participant mismatch: {path}')
    frame = frame.set_index('participant_id').loc[list(test.participant_id)].reset_index()
    values = frame[['target', score_column]].to_numpy(float)
    if not np.isfinite(values).all() or not np.array_equal(values[:, 0], test[endpoint].to_numpy(float)):
        raise ValueError(f'Prediction target mismatch or nonfinite values: {path}')
    if not np.isin(values[:, 0], [0, 1]).all() or not ((values[:, 1] >= 0) & (values[:, 1] <= 1)).all():
        raise ValueError(f'Invalid binary probabilities/targets: {path}')
    return frame[['participant_id', 'target', score_column]].rename(columns={score_column: 'probability'})


def reusable_reference(reference):
    """Conservative input/loss audit: never reuse the historical ANYPLAQ runs."""
    o = reference['options']
    checks = {
        'completed': reference['status'].get('status') == 'completed',
        'best_checkpoint': (reference['run'] / 'best_vae_net.pth').is_file(),
        'age_conditioning': flag(o, 'use_cond') and value(o, 'encoder_cond_mode', 'none') == 'demo',
        'no_weight': not flag(o, 'cond_weight'),
        'no_disease_conditioning': not flag(o, 'cond_athero', True) and not flag(o, 'cond_plaque'),
        'no_other_supervised_heads': not flag(o, 'aux_predict_athero', True)
             and not flag(o, 'aux_predict_plaque') and not flag(o, 'aux_predict_regression'),
        'no_conditional_prior': not flag(o, 'use_cond_prior'),
        'unweighted_BCE': float(value(o, 'aux_binary_pos_weight', 1)) == 1,
        'no_disease_sampling': not flag(o, 'balance_athero'),
        'gaussian_KL': value(o, 'latent_prior', 'gaussian') == 'gaussian'
             and value(o, 'prior_match_mode', 'auto') in ('kl', 'auto'),
        'prediction_file': (reference['study'] / 'heldout' / 'predictions.csv').is_file(),
        'not_resumed': not flag(o, 'continue_train'),
    }
    return all(checks.values()), checks


def immutable_text(path, text):
    path = Path(path)
    text = text.replace('\r\n', '\n')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError(f'Existing prepared inputs differ: {path}. Use a new --run-tag/output-dir.')
    if not path.exists():
        path.write_text(text, encoding='utf-8', newline='\n')


def training_command(reference, name, paths, group, variant, checkpoints, root=ROOT):
    o = {key: list(val) for key, val in reference['options'].items()}
    for key in ('clinical_feature_cols', 'continue_train'):
        o.pop('--' + key, None)
    for key, setting in {'name': name, 'checkpoints_dir': checkpoints, 'cond_tsv': paths['train'],
                         'val_cond_tsv': paths['validation'], 'feature_source': 'scan',
                         'aux_binary_col': paths['endpoint'], 'aux_binary_pos_weight': 1,
                         'latent_prior': 'gaussian', 'prior_match_mode': 'kl',
                         'encoder_cond_mode': 'demo' if variant == 'scans_age' else 'none',
                         'cond_stregkode_col': 'participant_id', 'cond_age_col': 'study_age',
                         'cond_age_source': 'age',
                         'cond_gender_col': 'predictor_gender' if group == 'combined' else 'study_gender'}.items():
        set_value(o, key, setting)
    for key in ('cond_weight', 'cond_athero', 'cond_plaque', 'aux_predict_athero', 'aux_predict_plaque',
                'aux_predict_regression', 'use_cond_prior', 'balance_athero', 'encoder_use_cond'):
        set_flag(o, key, False)
    set_flag(o, 'use_cond', variant == 'scans_age')
    set_flag(o, 'aux_predict_binary', True)
    if group == 'combined':
        o.pop('--only_gender', None)
    else:
        set_value(o, 'only_gender', group)
    command = [sys.executable, str(Path(root) / 'scripts' / 'training' / 'train_mesh_vae.py')]
    for key, values in o.items():
        command.extend([key, *values])
    return command


def evaluation_command(train_command, test_path, root=ROOT):
    o = command_options(train_command)
    command = [sys.executable, str(Path(root) / 'scripts' / 'diagnostics' / 'diagnose_mesh_binary.py')]
    for key in ('name', 'checkpoints_dir', 'dataroot', 'only_gender', 'seed', 'batch_size',
                'cond_stregkode_col', 'cond_age_col', 'cond_gender_col'):
        if '--' + key in o:
            command.extend(['--' + key, *o['--' + key]])
    command.extend(['--cond_tsv', str(test_path), '--gpu_ids', '0', '--which_epoch', 'best'])
    return command


def prepare(args):
    import pandas as pd
    output, checkpoints = args.output_dir.resolve(), args.checkpoints_dir.resolve()
    targets = (['peripheral_total'] if args.targets == 'main' else list(TERRITORIES)
               if args.targets == 'territories' else ['peripheral_total', *TERRITORIES])
    groups = args.group or ['F', 'M', 'combined']
    plan = {'version': 1, 'run_tag': args.run_tag, 'reference_tag': args.reference_tag,
            'combined_policy': args.combined_policy, 'output_dir': str(output),
            'checkpoints_dir': str(checkpoints), 'cohorts': [], 'tasks': [],
            'interpretation': 'Same held-out endpoint/participants within each target/group. Dedicated combined fits omit sex as a predictor.',
            'warnings': ['Eligibility is inherited from saved reference splits, including their original age/weight completeness rules.',
                         'The seven territory endpoints include left/right/total summaries; they are not seven independent vascular beds.']}
    for target in targets:
        endpoint = 'presence_' + target
        references, errors = {}, {}
        for sex in ('F', 'M'):
            try:
                references[sex] = load_reference(checkpoints, target, sex, args.reference_tag)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                errors[sex] = str(exc)
        for group in groups:
            base = output / target / group
            cohort = {'target': target, 'group': group, 'endpoint': endpoint,
                      'splits_dir': str(base / 'splits'), 'predictions_dir': str(base / 'predictions'),
                      'endpoint_definition': 'Any observed plaque (>0 volume) in the saved reference endpoint; peripheral_total excludes coronary.',
                      'status': 'prepared'}
            plan['cohorts'].append(cohort)
            try:
                needed = ('F', 'M') if group == 'combined' else (group,)
                if any(sex in errors for sex in needed):
                    raise ValueError('; '.join(errors[sex] for sex in needed if sex in errors))
                reference = references[needed[0]]
                if group == 'combined':
                    same_settings(references['F'], references['M'])
                    frames = combine_frames(references['F']['frames'], references['M']['frames'], endpoint)
                else:
                    frames = reference['frames']
                paths = {'endpoint': endpoint}
                for split in SPLITS:
                    columns = ['participant_id', 'study_age', 'study_gender', endpoint]
                    if target == 'peripheral_total' and 'volume_peripheral_total' in frames[split]:
                        # Reporting metadata for the measured-volume definition check.
                        # No model input flag reads this column.
                        columns.append('volume_peripheral_total')
                    frame = frames[split][columns].copy()
                    if group == 'combined':
                        frame['predictor_gender'] = 'F'
                    path = base / 'splits' / f'{split}.tsv'
                    immutable_text(path, frame.to_csv(sep='\t', index=False))
                    paths[split] = str(path)
                cohort['counts'] = {split: len(frames[split]) for split in SPLITS}
                cohort['members'] = {split: list(frames[split].participant_id) for split in SPLITS}
                cohort['reference_runs'] = [str(references[sex]['run']) for sex in needed]
                cohort['source_sha256'] = {key: val for sex in needed for key, val in references[sex]['source_sha256'].items()}
                cohort['split_sha256'] = {paths[split]: sha256(paths[split]) for split in SPLITS}
                cohort['sex_predictor'] = 'absent; predictor_gender is constant F, true study_gender retained' if group == 'combined' else 'constant within the sex-specific cohort'
                cohort['prediction_paths'] = {variant: str(base / 'predictions' / f'{variant}.csv')
                                              for variant in ('age_only', *VARIANTS)}
                if group == 'combined' and args.combined_policy == 'pooled':
                    cohort['status'] = 'pooled_from_sex_models'
                    continue
                prediction, age_model = age_predictions(frames['train'], frames['test'], endpoint)
                prediction['group'] = group
                immutable_text(base / 'predictions' / 'age_only.csv', prediction.to_csv(index=False))
                write_json(base / 'age_model.json', age_model)
                for variant in VARIANTS:
                    name = f'GEFION_CLEAN_{target}_{variant}_{group}_{args.run_tag}_seed42'
                    command = training_command(reference, name, paths, group, variant, checkpoints)
                    task = {'task_index': len(plan['tasks']), 'target': target, 'endpoint': endpoint,
                            'group': group, 'variant': variant, 'run_name': name,
                            'run_dir': str(checkpoints / name), 'cohort_dir': str(base),
                            'train_command': command, 'evaluate_command': evaluation_command(command, paths['test']),
                            'predictions_path': str(base / 'predictions' / f'{variant}.csv'),
                            'status': 'pending', 'seed': 42, 'configured_epochs': 150,
                            'training_inputs': ['scans'] if variant == 'scans_only' else ['scans', 'age'],
                            'training_labels': [endpoint], 'reuse_audit': None}
                    if group != 'combined' and variant == 'scans_age':
                        reusable, audit = reusable_reference(reference)
                        task['reuse_audit'] = audit
                        if reusable:
                            reused = exact_predictions(reference['study'] / 'heldout' / 'predictions.csv', frames['test'], endpoint, 'meshvae')
                            reused['group'] = group
                            immutable_text(task['predictions_path'], reused.to_csv(index=False))
                            task.update(status='reused', run_name=reference['name'], run_dir=str(reference['run']),
                                        reference_predictions_sha256=sha256(reference['study'] / 'heldout' / 'predictions.csv'))
                    task['fingerprint'] = fingerprint({key: task[key] for key in ('target', 'group', 'variant', 'train_command', 'evaluate_command')}
                                                     | {'split_sha256': cohort['split_sha256']})
                    plan['tasks'].append(task)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                cohort.update(status='unavailable', reason=str(exc))
                print(f'[UNAVAILABLE] {target}/{group}: {exc}', file=sys.stderr)
    plan_path = output / 'plan.json'
    if plan_path.exists():
        prior = read_json(plan_path)
        if prior.get('run_tag') != plan['run_tag'] or prior.get('combined_policy') != plan['combined_policy']:
            raise ValueError('Existing plan has a different run tag/policy; use a new output directory')
        old = {task['task_index']: task for task in prior.get('tasks', [])}
        for task in plan['tasks']:
            if task['task_index'] in old and old[task['task_index']].get('fingerprint') != task['fingerprint']:
                raise ValueError('Existing task inputs/settings differ; use a new output directory/run tag')
            if task['task_index'] in old and old[task['task_index']].get('status') == 'completed':
                task['status'] = 'completed'
    write_json(plan_path, plan)
    print_plan(plan)
    return plan


def print_plan(plan):
    for task in plan['tasks']:
        print(f"{task['task_index']:2d} {task['target']:18s} {task['group']:8s} {task['variant']:11s} {effective_status(task)}")
    pending = [str(task['task_index']) for task in plan['tasks'] if effective_status(task) not in ('reused', 'completed')]
    print(f"[PLAN] {len(plan['tasks'])} VAE tasks; {len(pending)} require training. Age-only fits are CPU and already saved.")
    print('[ARRAY] ' + (','.join(pending) if pending else 'none'))
    print('[OUT] ' + plan['output_dir'])


def effective_status(task):
    path = Path(task['cohort_dir']) / f"{task['variant']}_status.json"
    state = task['status']
    if path.exists():
        status = read_json(path)
        if status.get('fingerprint') != task['fingerprint']:
            raise ValueError('Per-task status belongs to a different task definition')
        state = status.get('status', task['status'])
    if state == 'completed' and task.get('inference_check_required'):
        check_path = Path(task['cohort_dir']) / f"{task['variant']}_inference_check.json"
        check = read_json(check_path) if check_path.is_file() else {}
        if check.get('status') != 'passed' or check.get('fingerprint') != task['fingerprint']:
            return 'inference_check_pending'
    return state


def verify_cohort(cohort):
    import pandas as pd
    for path, expected in {**cohort['source_sha256'], **cohort['split_sha256']}.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError(f'Saved source/split changed after prepare: {path}')
    frames = {}
    for split in SPLITS:
        frame = pd.read_csv(Path(cohort['splits_dir']) / f'{split}.tsv', sep='\t', dtype={'participant_id': str})
        frame = validate_frame(frame, cohort['endpoint'], cohort['group'])
        if list(frame.participant_id) != cohort['members'][split]:
            raise ValueError('Prepared participant membership differs from the plan')
        if cohort['group'] == 'combined' and ('predictor_gender' not in frame or not (frame.predictor_gender == 'F').all()):
            raise ValueError('Combined model must not receive a sex predictor')
        frames[split] = frame
    validate_disjoint(frames)
    return frames


def audit(args):
    plan = read_json(args.output_dir / 'plan.json')
    for cohort in plan['cohorts']:
        if cohort['status'] == 'unavailable':
            print(f"[UNAVAILABLE] {cohort['target']}/{cohort['group']}: {cohort['reason']}")
            continue
        frames = verify_cohort(cohort)
        for variant, path in cohort['prediction_paths'].items():
            if Path(path).exists():
                exact_predictions(path, frames['test'], cohort['endpoint'])
        print(f"[OK] {cohort['target']}/{cohort['group']}: exact disjoint splits and valid saved predictions")
    print_plan(plan)


def diagnosis_predictions(run, test, endpoint):
    import pandas as pd
    summary = pd.read_csv(run / 'binary_summary.csv')
    if len(summary) != 1 or 'ResultsDir' not in summary:
        raise ValueError('Diagnosis must produce exactly one summary row with ResultsDir')
    if 'Experiment' in summary and str(summary.iloc[0]['Experiment']) != run.name:
        raise ValueError('Diagnosis summary belongs to another run')
    path = Path(str(summary.iloc[0]['ResultsDir'])) / 'predictions.csv'
    if not path.is_file():
        path = run / 'binary_diagnostics' / Path(str(summary.iloc[0]['ResultsDir'])).name / 'predictions.csv'
    return exact_predictions(path, test, endpoint)


def train(args):
    plan_path = args.output_dir / 'plan.json'
    plan = read_json(plan_path)
    if args.task_index is None or not 0 <= args.task_index < len(plan['tasks']):
        raise ValueError(f'--task-index must be in 0..{len(plan["tasks"]) - 1}')
    task = plan['tasks'][args.task_index]
    cohort = next(row for row in plan['cohorts'] if row['target'] == task['target'] and row['group'] == task['group'])
    frames = verify_cohort(cohort)
    if effective_status(task) in ('completed', 'reused'):
        exact_predictions(task['predictions_path'], frames['test'], task['endpoint'])
        print(f"[SKIP] Task {task['task_index']} is {effective_status(task)}; held-out predictions verified.")
        return
    run = Path(task['run_dir'])
    done = run / 'clean_comparison_training_completed.json'
    status_path = Path(task['cohort_dir']) / f"{task['variant']}_status.json"
    if done.exists():
        if read_json(done).get('fingerprint') != task['fingerprint']:
            raise ValueError('Saved training marker belongs to different inputs/settings')
        if not (run / 'best_vae_net.pth').is_file():
            raise ValueError('Training completion marker exists without its checkpoint')
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError(f'Partial or unrecognized existing run: {run}. Do not overwrite; use a new run tag.')
        print('[TRAIN] ' + shlex.join(task['train_command']), flush=True)
        write_json(status_path, {'status': 'training', 'fingerprint': task['fingerprint']})
        subprocess.run(task['train_command'], check=True)
        if not (run / 'best_vae_net.pth').is_file() or not (run / 'training_summary.json').is_file():
            raise ValueError('Training exited without a best checkpoint and training summary')
        summary = read_json(run / 'training_summary.json')
        if (summary.get('train_samples') != len(frames['train'])
                or summary.get('validation_samples') != len(frames['validation'])
                or set(summary.get('validation_ids', [])) != set(frames['validation'].participant_id)):
            raise ValueError('Training loader participants differ from the prepared splits')
        write_json(done, {'status': 'completed', 'fingerprint': task['fingerprint']})
    print('[EVALUATE] ' + shlex.join(task['evaluate_command']), flush=True)
    write_json(status_path, {'status': 'evaluating', 'fingerprint': task['fingerprint']})
    subprocess.run(task['evaluate_command'], check=True)
    prediction = diagnosis_predictions(run, frames['test'], task['endpoint'])
    prediction['group'] = task['group']
    immutable_text(task['predictions_path'], prediction.to_csv(index=False))
    write_json(status_path, {'status': 'completed', 'fingerprint': task['fingerprint'],
                             'predictions_path': task['predictions_path'], 'run_dir': str(run)})
    # Per-task status files avoid array workers racing to rewrite shared plan.json.
    print(f"[COMPLETED] task {task['task_index']}: {task['predictions_path']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'audit', 'train'])
    parser.add_argument('--checkpoints-dir', type=Path, default=CHECKPOINTS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'clean_comparison')
    parser.add_argument('--reference-tag', default='next_v1')
    parser.add_argument('--run-tag', default='clean_v1')
    parser.add_argument('--targets', choices=['main', 'territories', 'all'], default='all')
    parser.add_argument('--group', nargs='+', choices=['F', 'M', 'combined'])
    parser.add_argument('--combined-policy', choices=['dedicated', 'pooled'], default='dedicated',
                        help='Default trains one joint F/M model with sex held constant as a predictor.')
    parser.add_argument('--task-index', type=int)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_tag):
        parser.error('--run-tag may contain only letters, numbers, underscore and hyphen')
    try:
        {'prepare': prepare, 'audit': audit, 'train': train}[args.action](args)
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        raise SystemExit('[ERROR] ' + str(exc)) from exc


if __name__ == '__main__':
    main()
