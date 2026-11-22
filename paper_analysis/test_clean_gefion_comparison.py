"""CPU checks for matched data, predictor exclusion and immutable run dispatch."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location('clean_comparison', Path(__file__).with_name('clean_gefion_comparison.py'))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoints = self.root / 'checkpoints'
        self.output = self.root / 'clean_comparison'

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, target='peripheral_total', group='F', weight=False):
        endpoint = 'presence_' + target
        run = self.checkpoints / runner.reference_name(target, group, 'next_v1')
        study = run / 'study'
        study.mkdir(parents=True)
        frames = {}
        for index, split in enumerate(runner.SPLITS):
            frames[split] = pd.DataFrame({'participant_id': [f'{group}_{split}_{j}' for j in range(4)],
                'study_age': [30 + index, 40 + index, 55 + index, 70 + index],
                'study_gender': group, endpoint: [0, 0, 1, 1], 'study_weight_kg': [50, 60, 80, 100]})
            frames[split].to_csv(study / f'{split}.tsv', sep='\t', index=False)
        command = ['python', 'train_mesh_vae.py', '--name', run.name,
                   '--checkpoints_dir', str(self.checkpoints), '--dataroot', str(self.root / 'scans'),
                   '--cond_tsv', str(study / 'train.tsv'), '--val_cond_tsv', str(study / 'validation.tsv'),
                   '--only_gender', group, '--feature_source', 'scan', '--seed', '42', '--split_seed', '42',
                   '--niter', '150', '--niter_decay', '0', '--batch_size', '8', '--use_cond',
                   '--encoder_cond_mode', 'demo', '--cond_age_source', 'age', '--cond_age_col', 'study_age',
                   '--cond_gender_col', 'study_gender', '--age_bins', '18', '35', '50', '60', '200',
                   '--cond_weight' if weight else '--no-cond_weight', '--weight_col', 'study_weight_kg',
                   '--no-cond_athero', '--no-cond_plaque', '--no-aux_predict_athero',
                   '--no-aux_predict_plaque', '--no-aux_predict_regression', '--no-balance_athero',
                   '--aux_predict_binary', '--aux_binary_col', endpoint, '--aux_binary_pos_weight', '1']
        manifest = {'gender': group, 'experiment': {'territory': target, 'kind': 'binary'},
                    'config': {'run_tag': 'next_v1', 'seed': 42, 'niter': 150, 'niter_decay': 0, 'batch_size': 8},
                    'members': {split: list(frame.participant_id) for split, frame in frames.items()}}
        runner.write_json(study / 'manifest.json', manifest)
        runner.write_json(study / 'status.json', {'status': 'completed'})
        runner.write_json(study / 'train_command.json', command)
        (run / 'best_vae_net.pth').write_bytes(b'fixture only')
        (study / 'heldout').mkdir()
        pd.DataFrame({'participant_id': frames['test'].participant_id,
                      'target': frames['test'][endpoint], 'meshvae': [.1, .2, .7, .8]}).to_csv(study / 'heldout' / 'predictions.csv', index=False)
        return run, frames

    def args(self, targets='main', **values):
        settings = dict(output_dir=self.output, checkpoints_dir=self.checkpoints, targets=targets,
                        group=None, combined_policy='dedicated', reference_tag='next_v1', run_tag='clean_v1')
        settings.update(values)
        return argparse.Namespace(**settings)

    def prepare(self, **values):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runner.prepare(self.args(**values))

    def test_dedicated_combined_and_missing_scan_fits(self):
        self.fixture(group='F')
        self.fixture(group='M')
        plan = self.prepare()
        self.assertEqual(len(plan['tasks']), 6)
        self.assertEqual(sum(task['status'] == 'reused' for task in plan['tasks']), 2)
        joint = [task for task in plan['tasks'] if task['group'] == 'combined']
        self.assertEqual(len(joint), 2)
        for task in joint:
            options = runner.command_options(task['train_command'])
            self.assertNotIn('--only_gender', options)
            self.assertEqual(runner.value(options, 'cond_gender_col'), 'predictor_gender')
            self.assertFalse(runner.flag(options, 'cond_weight'))
            self.assertFalse(runner.flag(options, 'use_cond_prior'))
        cohort = next(row for row in plan['cohorts'] if row['group'] == 'combined')
        test = pd.read_csv(Path(cohort['splits_dir']) / 'test.tsv', sep='\t')
        self.assertEqual(set(test.study_gender), {'F', 'M'})
        self.assertEqual(set(test.predictor_gender), {'F'})
        self.assertEqual(len(test), 8)
        self.assertTrue(Path(cohort['prediction_paths']['age_only']).is_file())
        scan = runner.command_options(joint[0]['train_command'])
        self.assertFalse(runner.flag(scan, 'use_cond'))
        self.assertEqual(runner.value(scan, 'encoder_cond_mode'), 'none')
        self.assertFalse(any('paper_analysis' in name for name in runner.__dict__ if name.startswith('import')))

    def test_weighted_reference_is_not_reused(self):
        self.fixture(group='F', weight=True)
        self.fixture(group='M', weight=True)
        plan = self.prepare()
        self.assertEqual(sum(task['status'] == 'reused' for task in plan['tasks']), 0)
        for task in plan['tasks']:
            self.assertFalse(runner.flag(runner.command_options(task['train_command']), 'cond_weight'))

    def test_generated_commands_use_actual_train_and_test_parsers(self):
        from meshvae.options import TrainOptions, TestOptions
        self.fixture(group='F')
        self.fixture(group='M')
        plan = self.prepare()
        for task in plan['tasks']:
            train = TrainOptions().parser.parse_args(task['train_command'][2:])
            test = TestOptions().parser.parse_args(task['evaluate_command'][2:])
            self.assertFalse(train.cond_weight)
            self.assertFalse(train.cond_athero)
            self.assertFalse(train.cond_plaque)
            self.assertFalse(train.use_cond_prior)
            self.assertFalse(train.aux_predict_athero)
            self.assertFalse(train.aux_predict_plaque)
            self.assertEqual(train.aux_binary_pos_weight, 1)
            self.assertEqual(train.aux_binary_col, 'presence_peripheral_total')
            self.assertEqual(test.name, train.name)
            self.assertEqual(test.cond_gender_col, train.cond_gender_col)
            if task['group'] == 'combined':
                self.assertIsNone(train.only_gender)
                self.assertIsNone(test.only_gender)

    def test_optional_main_volume_is_report_only(self):
        for group in ('F', 'M'):
            run, frames = self.fixture(group=group)
            for split in runner.SPLITS:
                frame = frames[split]
                frame['volume_peripheral_total'] = [0., 0., 10., np.nan]
                frame.to_csv(run / 'study' / f'{split}.tsv', sep='\t', index=False)
        plan = self.prepare()
        for cohort in plan['cohorts']:
            test = pd.read_csv(Path(cohort['splits_dir']) / 'test.tsv', sep='\t')
            self.assertIn('volume_peripheral_total', test)
            self.assertTrue(test.volume_peripheral_total.isna().any())
        for task in plan['tasks']:
            self.assertNotIn('volume_peripheral_total', task['train_command'])

    def test_full_plan_counts_only_verified_main_reuse(self):
        for target in ('peripheral_total', *runner.TERRITORIES):
            for group in ('F', 'M'):
                self.fixture(target, group, weight=target != 'peripheral_total')
        plan = self.prepare(targets='all')
        self.assertEqual(len(plan['tasks']), 48)
        self.assertEqual(sum(task['status'] == 'reused' for task in plan['tasks']), 2)
        self.assertEqual(len(plan['cohorts']), 24)

    def test_endpoint_mismatch_rejected_before_training(self):
        run, _ = self.fixture()
        path = run / 'study' / 'train_command.json'
        command = runner.read_json(path)
        command[command.index('--aux_binary_col') + 1] = 'benchmark_anyplaq'
        runner.write_json(path, command)
        with self.assertRaisesRegex(ValueError, 'another endpoint'):
            runner.load_reference(self.checkpoints, 'peripheral_total', 'F', 'next_v1')

    def test_joint_cross_split_overlap_rejected(self):
        _, female = self.fixture(group='F')
        _, male = self.fixture(group='M')
        male['test'].loc[0, 'participant_id'] = female['train'].iloc[0].participant_id
        with self.assertRaisesRegex(ValueError, 'overlap across sexes'):
            runner.combine_frames(female, male, 'presence_peripheral_total')

    def test_age_fit_does_not_use_heldout_labels_or_age_scaling(self):
        _, frames = self.fixture()
        prediction, model = runner.age_predictions(frames['train'], frames['test'], 'presence_peripheral_total')
        altered = frames['test'].copy()
        altered['presence_peripheral_total'] = 1 - altered.presence_peripheral_total
        second, second_model = runner.age_predictions(frames['train'], altered, 'presence_peripheral_total')
        np.testing.assert_allclose(prediction.probability, second.probability)
        self.assertEqual(model, second_model)
        self.assertEqual(model['training_age_mean'], frames['train'].study_age.mean())
        self.assertNotEqual(model['training_age_mean'], frames['test'].study_age.mean())

    def test_tampered_split_and_wrong_prediction_targets_rejected(self):
        self.fixture(group='F')
        self.fixture(group='M')
        plan = self.prepare()
        cohort = plan['cohorts'][0]
        frames = runner.verify_cohort(cohort)
        path = Path(cohort['prediction_paths']['age_only'])
        predictions = pd.read_csv(path)
        predictions['target'] = 1 - predictions.target
        predictions.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'target mismatch'):
            runner.exact_predictions(path, frames['test'], cohort['endpoint'])
        split = Path(cohort['splits_dir']) / 'train.tsv'
        split.write_text(split.read_text() + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed after prepare'):
            runner.verify_cohort(cohort)

    def test_idempotent_prepare_and_partial_run_protection(self):
        self.fixture(group='F')
        self.fixture(group='M')
        first = self.prepare()
        second = self.prepare()
        self.assertEqual(first, second)
        task = first['tasks'][0]
        run = Path(task['run_dir'])
        run.mkdir()
        (run / 'latest_vae_net.pth').write_bytes(b'partial fixture')
        args = self.args(task_index=0)
        with patch.object(runner.subprocess, 'run') as process:
            with self.assertRaisesRegex(ValueError, 'Partial or unrecognized'):
                runner.train(args)
            process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
