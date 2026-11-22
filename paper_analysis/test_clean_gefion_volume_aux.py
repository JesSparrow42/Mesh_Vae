"""CPU checks of matched volume supervision and existing public CLI contracts."""
from __future__ import annotations
import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = load('volume_comparison', 'clean_gefion_comparison.py')
volume = load('volume_aux', 'clean_gefion_volume_aux.py')


class VolumeTests(unittest.TestCase):
    def frames(self, group='F'):
        return {split: pd.DataFrame({'participant_id': [f'{group}_{split}_{i}' for i in range(5)],
            'study_age': [30., 40., 50., 60., 70.], 'study_gender': group,
            volume.ENDPOINT: [0, 0, 1, 1, 1], volume.RAW_VOLUME: [0., 0., 10., 20., np.nan]})
                for split in comparison.SPLITS}

    def test_training_only_statistics_and_no_test_volume_requirement(self):
        frames = self.frames()
        frames['validation'][volume.RAW_VOLUME] = [0., 0., 100., 200., np.nan]
        frames['test'][volume.RAW_VOLUME] = np.nan
        prepared, transform = volume.prepare_frames(frames, 'F', comparison)
        self.assertEqual(len(prepared['train']), 4)
        self.assertEqual(len(prepared['validation']), 4)
        self.assertEqual(len(prepared['test']), 5)
        x = np.log1p([0., 0., 10., 20.])
        self.assertAlmostEqual(transform['mean'], x.mean())
        self.assertAlmostEqual(transform['scale'], x.std())
        np.testing.assert_allclose(prepared['train'][volume.AUX_COLUMN].mean(), 0., atol=1e-12)
        self.assertTrue((prepared['test'][volume.AUX_COLUMN] == 0).all())
        self.assertTrue(prepared['test'][volume.RAW_VOLUME].isna().all())

    def test_invalid_volume_or_inconsistent_presence_rejected(self):
        for value in (-1., 0.):
            frames = self.frames()
            frames['train'].loc[2, volume.RAW_VOLUME] = value
            with self.assertRaises(ValueError):
                volume.prepare_frames(frames, 'F', comparison)

    def plan(self, root):
        cohorts, tasks = [], []
        for group in ('F', 'M', 'combined'):
            frames = self.frames(group if group != 'combined' else 'F')
            if group == 'combined':
                male = self.frames('M')
                frames = {s: pd.concat([frames[s], male[s]], ignore_index=True) for s in comparison.SPLITS}
                for frame in frames.values():
                    frame['predictor_gender'] = 'F'
            base = root/'peripheral_total'/group
            for split, frame in frames.items():
                comparison.immutable_text(base/'splits'/f'{split}.tsv', frame.to_csv(sep='\t', index=False))
            cohorts.append({'target': 'peripheral_total', 'endpoint': volume.ENDPOINT, 'group': group,
                'status': 'prepared', 'endpoint_definition': 'volume > 0', 'reference_runs': [],
                'source_sha256': {}, 'split_sha256': {}, 'sex_predictor': 'constant',
                'splits_dir': str(base/'splits'),
                'members': {s: list(f.participant_id) for s, f in frames.items()}})
            command = ['python', str(root/'train_mesh_vae.py'), '--name', 'original',
                       '--checkpoints_dir', str(root/'checkpoints'), '--feature_source', 'scan',
                       '--use_cond', '--encoder_cond_mode', 'demo', '--no-cond_plaque',
                       '--no-cond_athero', '--no-aux_predict_athero', '--aux_predict_binary',
                       '--aux_binary_col', volume.ENDPOINT, '--niter', '150', '--niter_decay', '0']
            tasks.append({'target': 'peripheral_total', 'group': group, 'variant': 'scans_age',
                          'configured_epochs': 150, 'train_command': command})
        return {'output_dir': str(root), 'checkpoints_dir': str(root/'checkpoints'),
                'run_tag': 'fixture', 'cohorts': cohorts, 'tasks': tasks}

    def test_six_tasks_exact_control_architecture_cli_and_idempotence(self):
        from meshvae.options import TrainOptions, TestOptions
        with tempfile.TemporaryDirectory() as folder:
            plan = self.plan(Path(folder))
            result = volume.append_volume_tasks(plan, comparison)
            self.assertEqual(len(result['tasks']), 9)
            again = volume.append_volume_tasks(result, comparison)
            self.assertEqual(len(again['tasks']), 9)
            self.assertEqual(len(again['cohorts']), 6)
            for group in ('F', 'M', 'combined'):
                selected = [t for t in result['tasks'] if t['target'] == volume.TARGET and t['group'] == group]
                self.assertEqual(len(selected), 2)
                settings = [comparison.command_options(t['train_command']) for t in selected]
                for options in settings:
                    options.pop('--name')
                    options.pop('--aux_plaque_weight')
                self.assertEqual(settings[0], settings[1])
                self.assertEqual([t['auxiliary_loss_weight'] for t in selected], [0., .25])
                for task in selected:
                    train = TrainOptions().parser.parse_args(task['train_command'][2:])
                    test = TestOptions().parser.parse_args(task['evaluate_command'][2:])
                    self.assertTrue(train.aux_predict_binary)
                    self.assertTrue(train.aux_predict_plaque)
                    self.assertFalse(train.aux_predict_regression)
                    self.assertFalse(train.cond_plaque)
                    self.assertEqual(train.aux_plaque_cols, [volume.AUX_COLUMN]*4)
                    self.assertEqual(train.aux_plaque_norm, 'none')
                    self.assertTrue(test.aux_predict_plaque)
                    self.assertEqual(test.aux_plaque_cols, [volume.AUX_COLUMN]*4)
                    self.assertEqual(test.aux_plaque_weight, 0)

    def test_inference_invariance_checks_probabilities_with_dummy_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            result = volume.append_volume_tasks(self.plan(Path(folder)), comparison)
            task = next(t for t in result['tasks'] if t['target'] == volume.TARGET)
            test = pd.read_csv(Path(task['cohort_dir'])/'splits'/'test.tsv', sep='\t')
            pred = pd.DataFrame({'participant_id': test.participant_id, 'target': test[volume.ENDPOINT],
                                 'probability': np.linspace(.1, .9, len(test))})
            comparison.immutable_text(task['predictions_path'], pred.to_csv(index=False))
            with patch.object(volume.subprocess, 'run') as run, patch.object(comparison, 'diagnosis_predictions', return_value=pred):
                volume.check_inference_invariance(task, comparison)
            changed = pd.read_csv(Path(task['cohort_dir'])/'splits'/'test_auxiliary_invariance.tsv', sep='\t')
            self.assertGreater(changed[volume.AUX_COLUMN].std(), 0)
            self.assertTrue(run.called)
            marker = comparison.read_json(Path(task['cohort_dir'])/(task['variant']+'_inference_check.json'))
            self.assertEqual(marker['status'], 'passed')
            self.assertAlmostEqual(marker['max_absolute_probability_difference'], 0.)


if __name__ == '__main__':
    unittest.main()
