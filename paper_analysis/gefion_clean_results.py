#!/usr/bin/env python3
"""Standalone Gefion comparison/figure workflow; upload this file only.

Bundled readable source is executed in isolated module namespaces. No files
are unpacked or installed, and existing training entry points are unchanged.
"""
import sys as _sys
import types as _types

def _component(name, source):
    module = _types.ModuleType(name)
    module.__file__ = __file__
    _sys.modules[name] = module
    exec(compile(source, __file__+":"+name, "exec"), module.__dict__)
    return module

COMPARISON = _component('_gefion_comparison', r'''#!/usr/bin/env python3
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
''')

LONG = _component('_gefion_long', r'''#!/usr/bin/env python3
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
''')

VOLUME = _component('_gefion_volume', r'''#!/usr/bin/env python3
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

''')

FIGURES = _component('_gefion_figures', r'''#!/usr/bin/env python3
"""Clean figures from the explicit matched-study plan; no training or project imports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

GROUPS = ("F", "M", "combined")
GROUP_LABELS = {"F": "Women", "M": "Men", "combined": "Combined model"}
COLORS = {"F": "#178D91", "M": "#D47560", "combined": "#7963A5"}
VARIANTS = ("age_only", "scans_only", "scans_age")
LABELS = {"age_only": "Age only", "scans_only": "Scans only", "scans_age": "Scans + age",
          "scans_age_plaque_aux": "Scans + age\n+ plaque supervision",
          "volume_control": "Classifier", "volume_aux": "Classifier\n+ volume supervision"}
VOLUME_TARGET = "peripheral_total_volume_training"
VOLUME_VARIANTS = ("volume_control", "volume_aux")
TERRITORIES = ("carotid_left", "carotid_right", "carotid_total", "femoral_left", "femoral_right", "femoral_total", "coronary_volume")
TITLES = {k: k.replace("_", " ").title() for k in TERRITORIES}
TITLES.update(coronary_volume="Coronary", peripheral_total="Overall peripheral plaque presence")
TITLES[VOLUME_TARGET] = "Does plaque-volume supervision help classification?"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path):
    data = pd.read_csv(path, sep="\t" if str(path).endswith(".tsv") else ",", dtype={"participant_id": str})
    if "participant_id" not in data or data.participant_id.isna().any() or data.participant_id.duplicated().any():
        raise ValueError(f"Expected unique nonempty participant IDs: {path}")
    return data.set_index("participant_id").sort_index()


def roc_auc(y, score):
    y, score = np.asarray(y, int), np.asarray(score, float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        raise ValueError("ROC AUC requires both outcome classes")
    order = np.argsort(score, kind="stable")
    _, start, counts = np.unique(score[order], return_index=True, return_counts=True)
    ranks = np.empty(len(y)); ranks[order] = np.repeat(start + (counts+1)/2, counts)
    return float((ranks[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def intervals(y, values, sex, count, seed=42):
    """Same sex-by-outcome stratified participant draws for every predictor arm."""
    y, sex = np.asarray(y, int), np.asarray(sex, str)
    strata = [np.flatnonzero((sex == s) & (y == c)) for s in sorted(set(sex)) for c in (0, 1)]
    strata = [g for g in strata if len(g)]
    keys = list(values); scores = np.column_stack([values[k] for k in keys])
    point = np.array([roc_auc(y, scores[:, i]) for i in range(len(keys))])
    draws = np.empty((count, len(keys))); rng = np.random.default_rng(seed)
    for b in range(count):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in strata])
        for i in range(len(keys)):
            draws[b, i] = roc_auc(y[ix], scores[ix, i])
    low, high = np.quantile(draws, [.025, .975], axis=0)
    rows = [{"Variant": k, "AUC": point[i], "CI95Lower": low[i], "CI95Upper": high[i]} for i, k in enumerate(keys)]
    comparisons = []
    for a, b in (("scans_age", "age_only"), ("scans_age", "scans_only"), ("volume_aux", "volume_control")):
        if a in keys and b in keys:
            i, j = keys.index(a), keys.index(b)
            lo, hi = np.quantile(draws[:, i]-draws[:, j], [.025, .975])
            comparisons.append({"Contrast": f"{a} minus {b}", "AUCDifference": point[i]-point[j], "CI95Lower": lo, "CI95Upper": hi})
    return rows, comparisons


def collect(root, bootstraps=5000):
    root = Path(root); plan_path = root/"plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("combined_policy", "dedicated") != "dedicated":
        raise ValueError("These requested figures require a model trained jointly on women and men")
    rows, contrasts, inventory, sources = [], [], [], [{"file": str(plan_path), "sha256": digest(plan_path)}]
    cohorts = plan.get("cohorts", plan.get("studies", []))
    if not cohorts:
        raise ValueError("The plan contains no explicitly prepared cohorts")
    study_keys = [(s["target"], s["group"]) for s in cohorts]
    if len(study_keys) != len(set(study_keys)):
        raise ValueError("Duplicate target/group studies in the explicit plan")
    for cohort in cohorts:
        target, group = cohort["target"], cohort["group"]
        base = root/target/group
        test_path = base/"splits"/"test.tsv"
        if not test_path.is_file():
            inventory.append({"Target": target, "Group": group, "Variant": "all", "Status": "missing_split"})
            continue
        test = read_table(test_path); col = cohort.get("endpoint", "presence_"+target)
        if col not in test or "study_gender" not in test:
            raise ValueError(f"Missing canonical target or true sex: {test_path}")
        y = pd.to_numeric(test[col], errors="coerce").to_numpy(float)
        if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError(f"Invalid binary targets: {test_path}")
        true_sex = test.study_gender.astype(str).to_numpy()
        if not set(true_sex).issubset({"F", "M"}) or (group != "combined" and not (true_sex == group).all()):
            raise ValueError(f"Incorrect true sex for {target}/{group}")
        sources.append({"file": str(test_path), "sha256": digest(test_path)})
        values = {}
        variants = VOLUME_VARIANTS if target == VOLUME_TARGET else VARIANTS
        for variant in variants:
            path = base/"predictions"/f"{variant}.csv"
            if not path.is_file():
                inventory.append({"Target": target, "Group": group, "Variant": variant, "Status": "missing"})
                continue
            if target == VOLUME_TARGET:
                task = next((t for t in plan["tasks"] if t["target"] == target and t["group"] == group and t["variant"] == variant), None)
                marker = base/f"{variant}_inference_check.json"
                check = json.loads(marker.read_text()) if marker.is_file() else {}
                if not task or check.get("status") != "passed" or check.get("fingerprint") != task["fingerprint"]:
                    inventory.append({"Target": target, "Group": group, "Variant": variant, "Status": "inference_check_pending"})
                    continue
                sources.append({"file": str(marker), "sha256": digest(marker)})
            pred = read_table(path)
            if set(pred.index) != set(test.index):
                raise ValueError(f"Prediction membership differs from saved test split: {path}")
            pred = pred.loc[test.index]
            if not {"target", "probability"}.issubset(pred):
                raise ValueError(f"Missing target/probability columns: {path}")
            p = pred[["target", "probability"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            if not np.isfinite(p).all() or not np.array_equal(p[:, 0], y) or ((p[:, 1] < 0) | (p[:, 1] > 1)).any():
                raise ValueError(f"Incorrect targets or probabilities: {path}")
            if "group" in pred and not (pred.group == group).all():
                raise ValueError(f"Incorrect model group: {path}")
            values[variant] = p[:, 1]
            sources.append({"file": str(path), "sha256": digest(path)})
            inventory.append({"Target": target, "Group": group, "Variant": variant, "Status": "available"})
        if values and len(set(y)) == 2:
            estimates, differences = intervals(y, values, true_sex, bootstraps)
            common = {"Target": target, "Group": group, "N": len(y), "Positive": int(y.sum())}
            rows += [{**common, **row} for row in estimates]
            contrasts += [{**common, **row} for row in differences]
        elif values:
            inventory.append({"Target": target, "Group": group, "Variant": "all", "Status": "AUC_undefined_one_class"})
    return pd.DataFrame(rows), pd.DataFrame(contrasts), pd.DataFrame(inventory), sources


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def panel(ax, table, target, variants, small=False):
    x = np.arange(len(variants)); width = .23
    for gi, group in enumerate(GROUPS):
        selected = table[(table.Target == target) & (table.Group == group)] if len(table) else pd.DataFrame()
        for j, variant in enumerate(variants):
            at = x[j]+(gi-1)*width
            row = selected[selected.Variant == variant] if len(selected) else pd.DataFrame()
            if row.empty:
                ax.text(at, .035, "Pending", rotation=90, ha="center", va="bottom", color="#7A8790", fontsize=7)
                continue
            if len(row) != 1:
                raise ValueError(f"Duplicate plot result {target}/{group}/{variant}")
            r = row.iloc[0]
            ax.bar(at, r.AUC, width*.94, color=COLORS[group])
            ax.vlines(at, r.CI95Lower, r.CI95Upper, colors="#223847", linewidth=1)
            ax.hlines([r.CI95Lower, r.CI95Upper], at-.025, at+.025, colors="#223847", linewidth=1)
            if not small:
                ax.text(at, max(r.CI95Upper, r.AUC)+.018, f"{r.AUC:.3f}", ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=[LABELS[v] for v in variants], ylim=(0, 1.075), yticks=np.arange(0, 1.01, .25))
    ax.axhline(.5, color="#9DA9B2", linewidth=.8, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5EAEE", linewidth=.6); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, labelsize=8 if small else 11)
    ax.tick_params(axis="y", length=0, labelsize=8 if small else 10)
    ax.set_title(TITLES.get(target, target), fontsize=11 if small else 17, pad=12)


def save(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(str(stem)+"."+suffix, dpi=300, facecolor="white", bbox_inches="tight")
    pyplot().close(fig)


def plot(table, output):
    plt = pyplot(); output = Path(output)
    from matplotlib.patches import Patch
    legend = [Patch(color=COLORS[g], label=GROUP_LABELS[g]) for g in GROUPS]
    footer = "95% participant bootstrap intervals. Combined = separately trained joint-sex model.\nAge-only = logistic baseline; each panel uses identical held-out participants across available predictors."
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    panel(ax, table, "peripheral_total", VARIANTS)
    ax.set_ylabel("ROC AUC"); fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .96), ncol=3, frameon=False)
    fig.text(.5, .02, footer, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .085, 1, .89)); save(fig, output/"01_main_auc")
    fig, axes = plt.subplots(2, 4, figsize=(14, 8.5))
    for ax, target in zip(axes.flat, TERRITORIES):
        panel(ax, table, target, VARIANTS, small=True); ax.set_ylabel("ROC AUC", fontsize=9)
    axes.flat[-1].axis("off"); axes.flat[-1].legend(handles=legend, loc="center", frameon=False, fontsize=11)
    fig.suptitle("Plaque presence by territory", fontsize=19)
    fig.text(.5, .015, footer+"\nTerritory-specific missing outcomes mean different territories may contain different participants.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .095, 1, .94), h_pad=3, w_pad=2); save(fig, output/"02_territory_auc")
    if len(table) and table.Target.eq(VOLUME_TARGET).any():
        fig, ax = plt.subplots(figsize=(9, 5.7))
        panel(ax, table, VOLUME_TARGET, VOLUME_VARIANTS)
        ax.set_ylabel("ROC AUC"); ax.set_title("Plaque volume as additional training supervision", fontsize=16)
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .96), ncol=3, frameon=False)
        fig.text(.5, .02, "Both classifiers use identical development participants with observed volume.\nAt test time the classifier receives scans and age only. Error bars: 95% paired participant bootstrap intervals.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .105, 1, .89)); save(fig, output/"03_volume_supervision_auc")


def volume_definition_check(study_dir, output):
    """Observed volume is the outcome definition, not an independent predictor."""
    rows = []
    for group in GROUPS:
        path = Path(study_dir)/"peripheral_total"/group/"splits"/"test.tsv"
        if not path.is_file():
            continue
        frame = read_table(path)
        columns = ["volume_peripheral_total", "presence_peripheral_total"]
        if not set(columns).issubset(frame):
            continue
        values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        keep = np.isfinite(values).all(axis=1) & (values[:, 0] >= 0)
        v, y = values[keep, 0], values[keep, 1]
        if not np.isin(y, [0, 1]).all() or not np.array_equal((v > 0).astype(float), y):
            raise ValueError(f"Volume/presence definition disagrees in {path}")
        score = roc_auc(y, (v > 0).astype(float)) if len(set(y)) == 2 else None
        rows.append({"Group": group, "N": len(y), "MissingVolume": int((~keep).sum()),
                     "AUC": score, "Meaning": "Deterministic target-definition check; not a trained prediction model"})
    table = pd.DataFrame(rows)
    table.to_csv(Path(output)/"volume_presence_definition.csv", index=False)
    if rows:
        plt = pyplot(); fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for i, row in enumerate(rows):
            if row["AUC"] is not None:
                ax.bar(i, row["AUC"], color=COLORS[row["Group"]], width=.55)
                ax.text(i, 1.02, f"1.000\nN = {row['N']}", ha="center", fontsize=12)
        ax.set(xticks=np.arange(len(rows)), xticklabels=["All participants" if r["Group"] == "combined" else GROUP_LABELS[r["Group"]] for r in rows],
               ylim=(0, 1.17), yticks=np.arange(0, 1.01, .25), ylabel="ROC AUC", title="Measured plaque volume determines plaque presence")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E5EAEE"); ax.set_axisbelow(True)
        fig.text(.5, .025, "Presence is defined as measured volume > 0. No model is fitted.\nThis is a definition check, not evidence of scan-model performance; missing volumes are excluded.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .12, 1, 1)); save(fig, Path(output)/"03_volume_presence_definition_check")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parents[1]/"clean_comparison")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--export-zip", type=Path)
    args = parser.parse_args(argv)
    if args.bootstraps < 100:
        parser.error("At least100 bootstrap draws required")
    output = args.output_dir or args.study_dir/"figures"; output.mkdir(parents=True, exist_ok=True)
    table, differences, inventory, sources = collect(args.study_dir, args.bootstraps)
    table.to_csv(output/"auc_results.csv", index=False); differences.to_csv(output/"paired_auc_differences.csv", index=False)
    inventory.to_csv(output/"availability.csv", index=False)
    (output/"figure_audit.json").write_text(json.dumps({"inputs": sources, "bootstraps": args.bootstraps, "seed": 42,
        "confidence_level": .95, "method": "sex-by-outcome stratified paired participant percentile bootstrap",
        "scope": "Conditional on fixed trained models; does not include training-seed variability",
        "pending": int(inventory.Status.eq("missing").sum()) if len(inventory) else 0}, indent=2), encoding="utf-8")
    plot(table, output)
    if args.export_zip:
        args.export_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.export_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in output.iterdir():
                if path.suffix in {".csv", ".json", ".png", ".pdf", ".svg"}:
                    archive.write(path, path.name)
    print(table.to_string(index=False) if len(table) else "No completed matched comparisons yet.")
    print(f"[OUT] {output.resolve()}")
    return table


if __name__ == "__main__":
    main()
''')

LOSSES = _component('_gefion_losses', r'''#!/usr/bin/env python3
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
''')

"""Entry point template embedded in the standalone Gefion delivery file."""
from pathlib import Path
import argparse
import json
import re
import shlex
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def settings(args):
    return argparse.Namespace(output_dir=args.study_dir.resolve(), checkpoints_dir=args.checkpoints_dir.resolve(),
        reference_tag=args.reference_tag, run_tag=args.run_tag, targets=args.targets, group=None,
        combined_policy="dedicated", task_index=args.task_index)


def prepare_study(args):
    plan = COMPARISON.prepare(settings(args))
    if args.include_long:
        plan = LONG.append_long_tasks(plan, COMPARISON)
    if args.include_volume:
        plan = VOLUME.append_volume_tasks(plan, COMPARISON)
    return plan


def preflight(plan, active_indices=()):
    for relative in ("scripts/training/train_mesh_vae.py", "scripts/diagnostics/diagnose_mesh_binary.py", "paper_analysis/next_steps_environment.sh"):
        if not (ROOT/relative).is_file():
            raise ValueError(f"Missing existing repository file: {ROOT/relative}")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from meshvae.options import TrainOptions, TestOptions
    train_parser, test_parser = TrainOptions().parser, TestOptions().parser
    for task in plan["tasks"]:
        train_parser.parse_args(task["train_command"][2:])
        test_parser.parse_args(task["evaluate_command"][2:])
        cohort = next(row for row in plan["cohorts"] if row["target"] == task["target"] and row["group"] == task["group"])
        COMPARISON.verify_cohort(cohort)
        run = Path(task["run_dir"])
        if (COMPARISON.effective_status(task) not in ("completed", "reused") and run.is_dir()
                and str(task["task_index"]) not in set(active_indices)
                and any(run.iterdir()) and not (run/"clean_comparison_training_completed.json").is_file()):
            raise ValueError(f"Partial training exists in {run}; it will not be overwritten. Inspect it before starting another run.")
        if task["variant"].startswith("long_") and "boson" in task["variant"]:
            for relative in ("scripts/diagnostics/check_boson_prior.py", "third_party/nonlinear_photonic_sampler"):
                if not (ROOT/relative).exists():
                    raise ValueError(f"Missing existing Boson sampler dependency: {ROOT/relative}")
    print("[PASS] All generated train/evaluate flags are supported; saved splits and partial-run safeguards verified.")


def sbatch(arguments):
    result = subprocess.run(["sbatch", "--parsable", *arguments], check=True, capture_output=True, text=True)
    job = result.stdout.strip().split(";")[0]
    if not re.fullmatch(r"\d+", job):
        raise ValueError(f"Unexpected scheduler response: {result.stdout.strip()}")
    return job


def submit_study(args):
    plan_path = args.study_dir/"plan.json"
    previous_submission = args.study_dir/"submission.json"
    active_indices, active_jobs, history = set(), [], []
    if previous_submission.is_file():
        previous = json.loads(previous_submission.read_text())
        history = previous.get("submissions", [{key: value for key, value in previous.items() if key != "submissions"}])
        for submission in history:
            job = submission["training_job"]
            result = subprocess.run(["squeue", "-h", "-j", job], capture_output=True, text=True)
            if result.returncode and "invalid job id" not in result.stderr.lower():
                raise ValueError(f"Cannot check existing job {job}: {result.stderr.strip()}")
            if result.stdout.strip():
                active_jobs.append(job)
                active_indices.update(str(i) for i in submission.get("task_indices", []))
    if plan_path.is_file():
        plan = COMPARISON.read_json(plan_path)
        if plan["run_tag"] != args.run_tag or Path(plan["checkpoints_dir"]).resolve() != args.checkpoints_dir.resolve():
            raise ValueError("Existing plan tag/checkpoint root differs from this command")
        if args.include_long:
            plan = LONG.append_long_tasks(plan, COMPARISON)
        if args.include_volume:
            plan = VOLUME.append_volume_tasks(plan, COMPARISON)
    else:
        plan = prepare_study(args)
    if not plan["tasks"]:
        raise ValueError("No trainable cohorts were prepared. Read the [UNAVAILABLE] reasons; no jobs were submitted.")
    preflight(plan, active_indices)
    indices = [str(task["task_index"]) for task in plan["tasks"] if COMPARISON.effective_status(task) not in ("completed", "reused")
               and str(task["task_index"]) not in active_indices]
    if not indices:
        if active_jobs:
            print(f"[ALREADY QUEUED] Training jobs {', '.join(active_jobs)}. No duplicate tasks submitted.")
            return
        print("[DONE] No missing VAE runs. Generating figures from completed results.")
        make_figures(args)
        return
    logs = args.study_dir/"logs"; logs.mkdir(parents=True, exist_ok=True)
    script = args.study_dir/"gefion_clean_array.sh"
    command = shlex.join([str(Path(__file__).resolve()), "train", "--study-dir", str(args.study_dir.resolve())])
    script.write_text("#!/bin/bash -l\nset -euo pipefail\n"+
        f"cd {shlex.quote(str(ROOT))}\nsource paper_analysis/next_steps_environment.sh gefion\n"+
        '"$STUDY_PYTHON" -c "import torch,open3d,trimesh; assert torch.cuda.is_available(), \'No allocated GPU visible\'; print(torch.cuda.get_device_name(0))"\n'+
        f'exec "$STUDY_PYTHON" {command} --task-index "${{SLURM_ARRAY_TASK_ID:?}}"\n', encoding="utf-8", newline="\n")
    job = sbatch(["--account=cu_0080", "--partition=defq", "--nodes=1", "--ntasks=1", "--gpus=1", "--cpus-per-task=8",
        "--mem=128G", "--time=3-00:00:00", f"--array={','.join(indices)}%{args.parallel}", "--job-name=clean_models",
        f"--output={logs}/train_%A_%a.out", f"--error={logs}/train_%A_%a.err", str(script)])
    submission = {"training_job": job, "task_indices": indices, "figure_job": None}
    COMPARISON.write_json(previous_submission, {**submission, "submissions": [*history, submission.copy()]})
    print(f"Submitted training array {job}: {len(indices)} missing fits; at most {args.parallel} run simultaneously.")
    collector = args.study_dir/"gefion_clean_figures.sh"
    collector.write_text("#!/bin/bash -l\nset -euo pipefail\n"+
        f"cd {shlex.quote(str(ROOT))}\nsource paper_analysis/next_steps_environment.sh gefion\n"+
        f'exec "$STUDY_PYTHON" '+shlex.join([str(Path(__file__).resolve()), "export", "--study-dir", str(args.study_dir.resolve()),
             "--checkpoints-dir", str(args.checkpoints_dir.resolve()), "--bootstraps", str(args.bootstraps)])+"\n", encoding="utf-8", newline="\n")
    try:
        figure_job = sbatch(["--account=cu_0080", "--partition=defq", "--nodes=1", "--ntasks=1", "--cpus-per-task=2",
            "--mem=16G", "--time=02:00:00", f"--dependency=afterany:{':'.join([*active_jobs, job])}", "--job-name=clean_figures",
            f"--output={logs}/figures_%j.out", f"--error={logs}/figures_%j.err", str(collector)])
        submission["figure_job"] = figure_job
        COMPARISON.write_json(previous_submission, {**submission, "submissions": [*history, submission.copy()]})
        print(f"Figure/export job {figure_job} will run when the training array finishes, including after failed tasks.")
    except subprocess.CalledProcessError as exc:
        print(f"[NOTE] Training is submitted, but the automatic figure job was rejected: {exc.stderr}")
        print("Run the same Python file with action 'export' after training finishes.")


def make_figures(args, export=False):
    # Earlier and newly appended arrays may both have dependent export jobs.
    # Linux releases this process lock automatically even after a failed job.
    args.study_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.study_dir/".figure_export.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return render_figures(args, export)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def render_figures(args, export=False):
    errors = []
    output = args.study_dir/"figures"
    try:
        FIGURES.main(["--study-dir", str(args.study_dir), "--output-dir", str(output), "--bootstraps", str(args.bootstraps)])
    except Exception as exc:
        errors.append("AUC figures: "+str(exc)); print("[ERROR] "+errors[-1], file=sys.stderr)
    try:
        LOSSES.main(["--checkpoints-dir", str(args.checkpoints_dir), "--output-dir", str(output/"loss_curves"),
            "--comparison-plan", str(args.study_dir/"plan.json")])
    except Exception as exc:
        errors.append("Loss figures: "+str(exc)); print("[ERROR] "+errors[-1], file=sys.stderr)
    if export:
        target = args.study_dir/"clean_figures.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in output.rglob("*"):
                # Exclude superseded definition-check charts from earlier exports.
                if (path.is_file() and path.suffix in {".csv", ".json", ".png", ".pdf", ".svg"}
                        and "volume_presence_definition" not in path.name
                        and "plaque_training_supervision" not in path.name):
                    archive.write(path, path.relative_to(output))
        print(f"[DOWNLOAD] {target.resolve()} (plots, aggregate metrics and provenance; no participant-level predictions)")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"[FIGURES] {output.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-file Gefion workflow: matched AUC figures and normal/Boson loss curves. No dependency installations.")
    parser.add_argument("action", choices=["prepare", "submit", "train", "status", "figures", "losses", "export"])
    parser.add_argument("--study-dir", type=Path, default=ROOT/"clean_comparison")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("/dcai/projects/cu_0080/results/checkpoints"))
    parser.add_argument("--reference-tag", default="next_v1")
    parser.add_argument("--run-tag", default="clean_v1")
    parser.add_argument("--targets", choices=["main", "territories", "all"], default="all")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--include-long", action="store_true", help="Also prepare six new matched 2000-epoch Gaussian/KL, Gaussian/MMD and Boson/MMD fits for women/men.")
    parser.add_argument("--include-volume", action="store_true", help="Also prepare six matched classifier-control / classifier+volume-supervision fits for women, men and combined.")
    args = parser.parse_args(argv)
    if args.parallel < 1 or args.bootstraps < 100:
        parser.error("--parallel must be positive; --bootstraps must be at least 100")
    args.study_dir = args.study_dir.resolve(); args.checkpoints_dir = args.checkpoints_dir.resolve()
    if args.action == "prepare":
        plan = prepare_study(args); preflight(plan)
    elif args.action == "submit":
        submit_study(args)
    elif args.action == "train":
        plan = COMPARISON.read_json(args.study_dir/"plan.json")
        if args.task_index is None or not 0 <= args.task_index < len(plan["tasks"]):
            raise ValueError("A valid --task-index is required for a training worker")
        task = plan["tasks"][args.task_index]
        try:
            COMPARISON.train(settings(args))
            if task["target"] == "peripheral_total_volume_training":
                VOLUME.check_inference_invariance(task, COMPARISON)
        except Exception as exc:
            COMPARISON.write_json(Path(task["cohort_dir"])/f"{task['variant']}_status.json",
                {"status": "failed", "fingerprint": task["fingerprint"], "reason": str(exc)})
            raise
    elif args.action == "status":
        COMPARISON.audit(settings(args))
        path = args.study_dir/"submission.json"
        if path.is_file():
            submission = COMPARISON.read_json(path)
            jobs = [row["training_job"] for row in submission.get("submissions", [submission])]
            try:
                subprocess.run(["sacct", "-X", "-j", ",".join(jobs), "--format=JobID,State,ExitCode,Elapsed"], check=False)
            except FileNotFoundError:
                print("[NOTE] sacct is unavailable here; the table above reports saved model status.")
    elif args.action in ("figures", "export"):
        make_figures(args, export=args.action == "export")
    elif args.action == "losses":
        LOSSES.main(["--checkpoints-dir", str(args.checkpoints_dir), "--output-dir", str(args.study_dir/"figures"/"loss_curves"),
            "--comparison-plan", str(args.study_dir/"plan.json")])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        detail = (exc.stderr or str(exc)) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise SystemExit("[ERROR] "+detail) from exc
