from pathlib import Path
import hashlib, json, os, re, sys, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
REPO = Path(os.environ.get('MQVAE_REPO', '/data/projects/3D_scans/ML/MQVAE_new')).expanduser().resolve()
if not (REPO / 'paper_analysis/collect_all_results.py').is_file():
    raise FileNotFoundError('Analysis code missing from {0}. Set MQVAE_REPO to the code folder.'.format(REPO))
sys.path.insert(0, str(REPO))
from paper_analysis.bootstrap_metrics import load_predictions, summarize_model
from paper_analysis.analyze_latent_signal import standardize_latent, choose_kmeans, make_tsne_figure
from paper_analysis.run_modality_benchmark import load_latent, load_metadata, merge_split
from paper_analysis.data_utils import read_table, numeric, resolve_id_column, clean_participant_id
import inspect
from typing import Any
from paper_analysis.data_utils import find_column, BMI_ALIASES, AGE_ALIASES, GENDER_ALIASES, normalize_gender, resolve_binary_target

def bmi_plot_data(frame: pd.DataFrame, bmi_column: str | None=None) -> pd.DataFrame:
    if bmi_column and bmi_column not in frame:
        raise KeyError('BMI column {0!r} is missing.'.format(bmi_column))
    source_column = bmi_column or find_column(frame, BMI_ALIASES)
    weight_column = find_column(frame, ('weight_kg', 'body_weight_kg', 'study_weight_kg'))
    height_column = find_column(frame, ('height_m', 'body_height_m', 'height_cm', 'body_height_cm', 'body_height_cm_3d'))
    result = pd.DataFrame(index=frame.index)
    result['bmi'] = numeric(frame[source_column]) if source_column else np.nan
    result['bmi_source_column'] = source_column or ''
    result['bmi_source_value'] = frame[source_column].astype(str) if source_column else ''
    result['bmi_weight_column'] = weight_column or ''
    result['bmi_height_column'] = height_column or ''
    result['bmi_weight_kg'] = numeric(frame[weight_column]) if weight_column else np.nan
    result['bmi_height_m'] = numeric(frame[height_column]) if height_column else np.nan
    if height_column and '_cm' in height_column.lower():
        result['bmi_height_m'] /= 100.0
    valid_components = np.isfinite(result['bmi_weight_kg']) & result['bmi_weight_kg'].gt(0) & np.isfinite(result['bmi_height_m']) & result['bmi_height_m'].gt(0)
    result['bmi_computed_check'] = (result['bmi_weight_kg'] / result['bmi_height_m'].pow(2)).where(valid_components)
    result['bmi_difference'] = result['bmi'] - result['bmi_computed_check']
    result['bmi_check_status'] = np.select([result['bmi_computed_check'].isna(), ~np.isfinite(result['bmi']), result['bmi_difference'].abs().gt(1.0)], ['unit-bearing components unavailable', 'source BMI unavailable', 'difference above 1 kg/m2'], default='within 1 kg/m2')
    return result

def tsne_parameters(rows: int, seed: int, perplexity: float) -> dict[str, Any]:
    from sklearn.manifold import TSNE
    parameters = {'n_components': 2, 'perplexity': min(float(perplexity), max(5.0, (rows - 1) / 3.0)), 'init': 'pca', 'learning_rate': 'auto', 'random_state': int(seed), 'verbose': 0}
    parameters['max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'] = 2000
    return parameters

def tsne_coordinate_fingerprint(values: np.ndarray, frame: pd.DataFrame, seed: int, perplexity: float) -> str:
    import sklearn
    matrix = np.ascontiguousarray(values, dtype='<f8')
    context = {'ordered_participants': list(zip(frame['participant_id'].astype(str), frame['split'].astype(str))), 'shape': matrix.shape, 'parameters': tsne_parameters(len(matrix), seed, perplexity), 'requested_perplexity': float(perplexity), 'sklearn_version': sklearn.__version__}
    digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode())
    digest.update(matrix.tobytes())
    return digest.hexdigest()

def make_tsne_figure(train_values: np.ndarray, test_values: np.ndarray, train: pd.DataFrame, test: pd.DataFrame, train_cluster: np.ndarray, test_cluster: np.ndarray, output_dir: Path, seed: int, perplexity: float, gender: str, representation_label: str='Latent representation', bmi_limits: tuple[float, float]=(15.0, 45.0), coordinates: np.ndarray | None=None) -> dict[str, float]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE, trustworthiness
    except ImportError:
        return {}
    values = np.concatenate([train_values, test_values], axis=0)
    frame = pd.concat([train.assign(split='train'), test.assign(split='test')], ignore_index=True)
    frame['cluster'] = np.concatenate([train_cluster, test_cluster])
    kwargs = tsne_parameters(len(values), seed, perplexity)
    effective_perplexity = kwargs['perplexity']
    if coordinates is None:
        coordinates = TSNE(**kwargs).fit_transform(values)
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(values), 2) or not np.isfinite(coordinates).all():
        raise ValueError('t-SNE coordinates must contain two finite values per participant.')
    frame['tsne_1'] = coordinates[:, 0]
    frame['tsne_2'] = coordinates[:, 1]
    (output_dir / 'tsne_coordinate_cache.json').write_text(json.dumps({'fingerprint': tsne_coordinate_fingerprint(values, frame, seed, perplexity), 'parameters': kwargs}, indent=2), encoding='utf-8')
    trust_n = min(3000, len(values))
    rng = np.random.default_rng(seed)
    trust_indices = rng.choice(len(values), size=trust_n, replace=False)
    trust = float(trustworthiness(values[trust_indices], coordinates[trust_indices], n_neighbors=min(15, max(2, trust_n // 20))))
    panels = 4 if 'plaque_volume' in frame else 3
    fig, axes_grid = plt.subplots(2, 2, figsize=(12.8, 10.2), constrained_layout=True)
    axes = axes_grid.ravel()
    x = frame['tsne_1'].to_numpy()
    y = frame['tsne_2'].to_numpy()
    plaque = frame['target'].to_numpy(dtype=int)
    for value, label, color in ((0, 'No measured plaque', '#0F6F73'), (1, 'Plaque present', '#D86D5A')):
        mask = plaque == value
        axes[0].scatter(x[mask], y[mask], s=10, alpha=0.58, color=color, linewidths=0, label='{0} (n={1})'.format(label, mask.sum()))
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title('A  Measured plaque presence')
    bmi = frame['bmi'].to_numpy(dtype=float)
    lower, upper = map(float, bmi_limits)
    if not np.isfinite([lower, upper]).all() or lower <= 0 or lower >= upper:
        raise ValueError('BMI color limits must be positive and increasing.')
    missing = np.isnan(bmi)
    invalid = ~missing & (~np.isfinite(bmi) | (bmi <= 0))
    below = np.isfinite(bmi) & (bmi > 0) & (bmi < lower)
    above = np.isfinite(bmi) & (bmi > upper)
    in_range = np.isfinite(bmi) & (bmi >= lower) & (bmi <= upper)
    frame['bmi_plot_status'] = np.select([missing, invalid, below, above], ['missing', 'invalid/nonpositive', 'below color scale', 'above color scale'], default='within color scale')
    color_points = axes[1].scatter(x[in_range], y[in_range], c=bmi[in_range], cmap='viridis', vmin=lower, vmax=upper, s=12, alpha=0.78, linewidths=0)
    for mask, label, marker, color in ((below, 'Below {0:g}'.format(lower), 'v', '#2463A3'), (above, 'Above {0:g}'.format(upper), '^', '#C3328B'), (missing, 'Missing', 'x', '#8A949A'), (invalid, 'Invalid/nonpositive', 'x', '#D23C30')):
        if mask.any():
            axes[1].scatter(x[mask], y[mask], s=28, marker=marker, color=color, linewidths=0.8, alpha=0.9, label='{0} (n={1})'.format(label, mask.sum()), zorder=4)
    if (below | above | missing | invalid).any():
        axes[1].legend(frameon=False, fontsize=8, loc='best')
    extension = 'both' if below.any() and above.any() else 'min' if below.any() else 'max' if above.any() else 'neither'
    fig.colorbar(color_points, ax=axes[1], label='BMI (kg/m²)', extend=extension)
    bmi_audit_columns = [column for column in frame if column == 'participant_id' or column == 'split' or column.startswith('bmi')]
    frame[bmi_audit_columns].to_csv(output_dir / 'bmi_audit.csv', index=False)
    bmi_summary = {'color_limits_kg_m2': [lower, upper], 'rows': len(frame), 'missing': int(missing.sum()), 'invalid': int(invalid.sum()), 'below_color_scale': int(below.sum()), 'above_color_scale': int(above.sum()), 'source_bmi_min': float(np.min(bmi[np.isfinite(bmi)])) if np.isfinite(bmi).any() else None, 'source_bmi_max': float(np.max(bmi[np.isfinite(bmi)])) if np.isfinite(bmi).any() else None, 'data_policy': 'Source BMI retained; out-of-scale points use separate markers; no unit conversion inferred.'}
    (output_dir / 'bmi_audit_summary.json').write_text(json.dumps(bmi_summary, indent=2), encoding='utf-8')
    frame.to_csv(output_dir / 'tsne_coordinates.csv', index=False)
    axes[1].set_title('B  Body mass index')
    clusters = frame['cluster'].to_numpy(dtype=int)
    for cluster in sorted(np.unique(clusters)):
        mask = clusters == cluster
        axes[2].scatter(x[mask], y[mask], s=10, alpha=0.62, linewidths=0, label='Cluster {0}'.format(cluster))
    axes[2].legend(frameon=False, fontsize=8, ncol=2)
    axes[2].set_title('C  Groups from training features')
    if panels == 4:
        volume = frame['plaque_volume'].to_numpy(dtype=float)
        valid_volume = np.isfinite(volume) & (volume >= 0)
        axes[3].scatter(x[~valid_volume], y[~valid_volume], s=8, alpha=0.35, color='#D5DDE1', linewidths=0)
        logged_volume = np.log1p(volume[valid_volume])
        maximum = max(float(logged_volume.max()) if len(logged_volume) else 0.0, 1.0)
        points = axes[3].scatter(x[valid_volume], y[valid_volume], c=logged_volume, cmap='magma', vmin=0.0, vmax=maximum, s=12, alpha=0.78, linewidths=0)
        colorbar = fig.colorbar(points, ax=axes[3], label='Measured volume (reported units; log1p scale)')
        ticks = np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
        ticks = ticks[np.log1p(ticks) <= maximum]
        colorbar.set_ticks(np.log1p(ticks), labels=['{0:g}'.format(tick) for tick in ticks])
        axes[3].set_title('D  Measured plaque volume')
    else:
        axes[3].set_visible(False)
    for axis in axes[:panels]:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel('t-SNE 1')
        axis.set_ylabel('t-SNE 2')
        axis.set_aspect('equal', adjustable='box')
        axis.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    fig.suptitle('{0}, {1}\n{2} training participants and {3} test participants'.format(representation_label, {'F': 'women', 'M': 'men', 'ALL': 'pooled cohort'}[gender], len(train), len(test)), fontsize=14)
    fig.savefig(output_dir / 'tsne_latent_signal.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'tsne_latent_signal.pdf', bbox_inches='tight')
    plt.close(fig)
    return {'tsne_perplexity': effective_perplexity, 'tsne_trustworthiness': trust}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
EXPORT_ROOT = Path(os.environ.get('MQVAE_EXPORT_DIR', str(REPO / 'model_results_export_20261013'))).expanduser().resolve()
EXPORT_DIRS = [EXPORT_ROOT / 'results_overview_computerome', EXPORT_ROOT / 'results_overview_gefion']
if not (EXPORT_DIRS[0] / 'all_results.csv').is_file():
    raise FileNotFoundError('Computerome export missing: {0}. Run collect_all_results.py with --include-plot-data first.'.format(EXPORT_DIRS[0] / 'all_results.csv'))
OUTPUT = EXPORT_ROOT / 'plots_recent'
OUTPUT.mkdir(parents=True, exist_ok=True)
REMOTE_TO_LOCAL = {}
INCLUDE_PROVISIONAL_REFERENCE = False
COMPUTE_MISSING_CIS = True
BOOTSTRAPS, CONFIDENCE, SEED = (5000, 0.95, 20261007)
PLOT_METRICS = ['auc', 'average_precision', 'r2', 'mae', 'spearman']
MAX_ROWS_PER_FIGURE = 30
RUN_TSNE, PERPLEXITY = (True, 30)
BMI_COLOR_LIMITS = (15.0, 45.0)
LATENT_JOBS = []
RECENT_RUN_TAGS = ['next_v1', 'next_boson_v1', 'best_boson_v1']
RECENT_SUITE_MODE = 'all_recent'
PLOT_METRICS = ['auc', 'mae', 'r2']
TSNE_PLATFORM = 'computerome'
TSNE_EXPERIMENT = 'plaque_all_log'
TSNE_RUN_TAG = 'next_v1'
TSNE_SEED = 42
MAX_TSNE_FIGURES = 2

def read_csv(path):
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
keys = ['all_results', 'confidence_intervals', 'result_inventory', 'plot_artifacts']
tables = {key: [] for key in keys}
input_status = []
for directory in EXPORT_DIRS:
    directory = Path(directory).expanduser().resolve()
    if not (directory / 'all_results.csv').is_file():
        input_status.append(dict(export=str(directory), status='missing', rows=0))
        continue
    for key in keys:
        if (directory / '{0}.csv'.format(key)).is_file():
            frame = read_csv(directory / '{0}.csv'.format(key))
            frame['_export_dir'] = str(directory)
            if key == 'plot_artifacts':
                frame['_artifact_dir'] = [str(Path(value).parent) if value and Path(value).is_file() else str(directory) for value in frame.get('export_file', pd.Series('', index=frame.index))]
            tables[key].append(frame)
    input_status.append(dict(export=str(directory), status='loaded', rows=len(tables['all_results'][-1])))
if not tables['all_results'] and INCLUDE_PROVISIONAL_REFERENCE:
    directory = REPO / 'outputs/model_overview/provisional_gefion'
    if (directory / 'all_results.csv').is_file():
        frame = read_csv(directory / 'all_results.csv')
        frame['_export_dir'] = str(directory)
        tables['all_results'].append(frame)
        input_status.append(dict(export=str(directory), status='provisional transcription only', rows=len(frame)))
results, saved_ci, inventory, artifacts = [pd.concat(tables[k], ignore_index=True).fillna('') if tables[k] else pd.DataFrame() for k in keys]
for field in ['platform', 'source_root', 'target', 'cohort', 'gender', 'seed', 'prior', 'task', 'target_transform', 'model', 'run', 'run_directory', 'evaluation_split', 'result_scope', 'evidence_level', 'preferred', 'source_file', 'source_row', 'reference_run', 'reference_fingerprint', 'reference_split_hashes']:
    if field not in results:
        results[field] = ''
results['_row_id'] = ['{0:06d}'.format(i) for i in range(len(results))]
preferred = results['preferred'].astype(str).str.lower().isin(['true', '1'])
provisional = results['evidence_level'].eq('transcribed_screenshot')
selected = results.loc[preferred | provisional & INCLUDE_PROVISIONAL_REFERENCE].copy()
tag_pattern = '(?:' + '|'.join((re.escape(tag) for tag in RECENT_RUN_TAGS)) + ')'
family_pattern = '_NEXT_(territory_classification|territory_volume|plaque|sensitivity|weight)_'
recent = selected['run'].str.contains(family_pattern.replace('(', '(?:', 1), regex=True) & selected['run'].str.contains('_(?:F|M)_' + tag_pattern + '_seed\\d+$', regex=True)
neural = selected['model'].str.lower().isin(['meshvae', 'mqvae', 'vae']) | selected['model'].str.lower().eq(selected['run'].str.lower())
selected['plot_family'] = selected['run'].str.extract(family_pattern, expand=False).fillna('')
in_scope = recent & neural & selected['result_scope'].eq('model_evaluation')
if RECENT_SUITE_MODE == 'primary':
    in_scope &= selected['plot_family'].isin(['plaque', 'weight'])
selection_audit = selected[['platform', 'source_root', 'run', 'model', 'target', 'cohort', 'gender', 'seed', 'source_file']].copy()
selection_audit['included'] = in_scope
selection_audit['reason'] = np.where(in_scope, 'recent NEXT model', 'outside selected tags/suites or a covariate baseline')
selected = selected.loc[in_scope].copy()
selected['original_model_label'] = selected['model']
selected['model'] = 'meshvae'
selected['_source_priority'] = np.where(selected['source_file'].str.replace('\\', '/', regex=False).str.contains('/study/heldout/', regex=False), 0, 1)
identity = ['platform', 'source_root', 'run', 'target', 'cohort', 'gender', 'seed', 'evaluation_split']
selected = selected.sort_values(['_source_priority', 'source_file', 'source_row'], kind='stable').drop_duplicates(identity, keep='first')
secondary = in_scope & ~selection_audit.index.isin(selected.index)
selection_audit.loc[secondary, 'included'] = False
selection_audit.loc[secondary, 'reason'] = 'secondary report of the same recent model'
selection_audit.to_csv(OUTPUT / 'recent_run_selection.csv', index=False)
display(pd.DataFrame(input_status))
if not inventory.empty:
    display(inventory.loc[~inventory.get('status', pd.Series('', index=inventory.index)).eq('collected')].head(100))
results.to_csv(OUTPUT / 'all_results_with_provenance.csv', index=False)
inventory.to_csv(OUTPUT / 'source_inventory.csv', index=False)
ALIASES = {'auc': ['auc', 'roc_auc', 'test_auc', 'test.roc_auc'], 'average_precision': ['averageprecision', 'average_precision', 'ap', 'test_average_precision', 'test_ap', 'test.average_precision'], 'r2': ['r2', 'test_r2', 'test.r2'], 'mae': ['mae', 'test_mae', 'test.mae'], 'rmse': ['rmse', 'test_rmse'], 'spearman': ['spearman', 'rho', 'test_spearman'], 'pearson': ['pearson', 'test_pearson'], 'balanced_accuracy': ['balancedaccuracy', 'balanced_accuracy', 'test_balanced_accuracy'], 'accuracy': ['accuracy', 'test_accuracy'], 'f1': ['f1', 'test_f1'], 'sensitivity': ['sensitivity', 'test_sensitivity'], 'specificity': ['specificity', 'test_specificity'], 'brier': ['brier'], 'transformed_r2': ['transformedr2', 'transformed_r2'], 'transformed_mae': ['transformedmae', 'transformed_mae'], 'transformed_rmse': ['transformedrmse', 'transformed_rmse'], 'transformed_pearson': ['transformedpearson', 'transformed_pearson'], 'transformed_spearman': ['transformedspearman', 'transformed_spearman']}

def metric_name(value):
    value = str(value).strip().lower()
    return next((name for name, aliases in ALIASES.items() if value in aliases), value)

def scalar(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def normalized_path(value):
    return str(value).replace('\\', '/').rstrip('/')

def artifact_path(row):
    return Path(row.get('_artifact_dir', row['_export_dir'])) / row['exported_file']

def local_artifact(original, export_dir=None):
    original = normalized_path(original)
    if not artifacts.empty and 'source_file' in artifacts:
        found = artifacts.loc[artifacts['source_file'].map(normalized_path).eq(original)]
        if export_dir is not None:
            found = found.loc[found['_export_dir'].eq(str(export_dir))]
        paths = {artifact_path(row) for _, row in found.iterrows()}
        paths = [path for path in paths if path.is_file()]
        if len(paths) == 1:
            return paths[0]
    for remote, local in sorted(REMOTE_TO_LOCAL.items(), key=lambda item: -len(item[0])):
        prefix = normalized_path(remote)
        if original == prefix or original.startswith(prefix + '/'):
            path = Path(local) / original[len(prefix):].lstrip('/')
            if path.is_file():
                return path
    path = Path(original).expanduser()
    return path if path.is_file() else None
metric_rows = []
for _, row in selected.iterrows():
    lower_keys = {str(key).lower(): key for key in row.index}
    metrics = {}
    for metric, aliases in ALIASES.items():
        if metric not in PLOT_METRICS:
            continue
        names = [alias + '_mean' for alias in aliases] if row['result_scope'] == 'seed_aggregate' else aliases
        values = [(scalar(row[lower_keys[name]]), lower_keys[name]) for name in names if name in lower_keys and np.isfinite(scalar(row[lower_keys[name]]))]
        if values:
            metrics[metric] = values[0]
    if row['result_scope'] == 'seed_paired_effect':
        metrics = {metric_name(row.get('Metric', '')): (scalar(row.get('MeanDirectionalImprovement')), 'MeanDirectionalImprovement')}
    for metric, (estimate, column) in metrics.items():
        if not np.isfinite(estimate):
            continue
        scale = 'transformed' if metric.startswith('transformed_') else 'original_target_units' if row['task'] in ['regression', 'hurdle'] else 'dimensionless'
        item = {**row.to_dict(), 'metric': metric, 'estimate': estimate, 'metric_scale': scale, 'original_metric_column': column, 'ci_low': np.nan, 'ci_high': np.nan, 'confidence_level': np.nan, 'ci_status': 'missing', 'ci_source': '', 'interval_method': '', 'resampling_unit': 'seed' if row['result_scope'].startswith('seed_') else 'participant', 'n_participants': scalar(row.get('N', row.get('n', row.get('n_participants', '')))), 'n_bootstrap_requested': np.nan, 'n_bootstrap_valid': np.nan}
        if row['result_scope'] == 'seed_paired_effect':
            item.update(ci_low=scalar(row.get('Improvement95CI_Low')), ci_high=scalar(row.get('Improvement95CI_High')), confidence_level=0.95, ci_status='saved_seed_interval', ci_source=row['source_file'], interval_method='paired-seed bootstrap; directional improvement')
        metric_rows.append(item)
tidy = pd.DataFrame(metric_rows)
linked_intervals = set()
if not tidy.empty and (not saved_ci.empty):
    for index, row in tidy.iterrows():
        try:
            ci_files = json.loads(row.get('ci_files') or '[]')
        except (json.JSONDecodeError, TypeError):
            ci_files = []
        if not ci_files or row['result_scope'] != 'model_evaluation':
            continue
        matches = saved_ci.loc[saved_ci['source_file'].isin(ci_files)].copy()
        if 'metric' not in matches:
            continue
        matches = matches.loc[matches['metric'].map(metric_name).eq(row['metric']) & matches['model'].eq(row['model'])]
        matches = matches.loc[matches.get('resampling_unit', pd.Series('', index=matches.index)).str.lower().eq('participant')]
        matches = matches.loc[~matches['model'].str.contains('minus', case=False)]
        bounds = {(scalar(ci.get('ci_low')), scalar(ci.get('ci_high'))) for _, ci in matches.iterrows()}
        bounds = {(lo, hi) for lo, hi in bounds if np.isfinite(lo) and np.isfinite(hi) and (lo <= hi)}
        if len(bounds) != 1:
            continue
        ci = matches.iloc[0]
        level = scalar(ci.get('confidence_level', ci.get('confidence', '')))
        if not np.isfinite(level) and '95' in str(ci.get('ci_source_field', '')):
            level = 0.95
        if not np.isclose(level, CONFIDENCE):
            continue
        lo, hi = next(iter(bounds))
        tidy.loc[index, ['ci_low', 'ci_high', 'confidence_level', 'ci_status', 'ci_source', 'interval_method']] = [lo, hi, level, 'saved_interval', ci['source_file'], ci.get('interval_method', 'saved participant bootstrap')]
        linked_intervals.update(matches.index)
ci_audit = []

def prediction_for(row):
    source = normalized_path(row['source_file'])
    folder = source.rsplit('/', 1)[0]
    candidates = []
    if '/study/heldout/' in source:
        candidates.append(folder + '/predictions.csv')
    for field in ['ResultsDir', 'results_dir']:
        value = row.get(field, '')
        if value:
            value = normalized_path(value)
            if not value.startswith('/') and (not re.match('^[A-Za-z]:/', value)):
                value = normalized_path(row['run_directory']) + '/' + value
            candidates.append(value + '/predictions.csv')
    candidates.extend([folder + '/test_predictions.csv', folder + '/predictions.csv'])
    for original in dict.fromkeys(candidates):
        path = local_artifact(original, row['_export_dir'])
        if path:
            return (original, path)
    return ('', None)
cache = OUTPUT / 'ci_cache'
cache.mkdir(exist_ok=True)
if COMPUTE_MISSING_CIS and (not tidy.empty):
    for row_id, group in tidy.groupby('_row_id', sort=False):
        row = group.iloc[0]
        if row['result_scope'] != 'model_evaluation' or row['evidence_level'] == 'transcribed_screenshot' or group['ci_low'].notna().all():
            continue
        entry = {'row_id': row_id, 'run': row['run'], 'model': row['model'], 'source_file': row['source_file']}
        original, path = prediction_for(row)
        if path is None:
            ci_audit.append({**entry, 'status': 'missing_predictions', 'reason': 'Collect with --include-plot-data or map server paths.'})
            continue
        try:
            source_frame = pd.read_csv(path, dtype=str)
            explicit_test_rows = 'split' in source_frame and source_frame['split'].astype(str).str.lower().isin(['test', 'heldout', 'held_out', 'held-out']).any()
            if str(row['evaluation_split']).lower() not in ['test', 'heldout', 'held_out', 'held-out'] and (not explicit_test_rows):
                ci_audit.append({**entry, 'status': 'unverified_split', 'reason': 'Requires explicit held-out provenance.'})
                continue
            if explicit_test_rows:
                tidy.loc[tidy['_row_id'].eq(row_id), 'evaluation_split'] = 'test'
            task = 'regression' if row['task'] == 'hurdle' else row['task']
            if task not in ['binary', 'regression']:
                raise ValueError('Unknown task {0!r}.'.format(task))
            signature = hashlib.sha256(path.read_bytes() + '{0}|{1}|{2}|{3}|{4}|{5}'.format(row['model'], row['target'], task, BOOTSTRAPS, CONFIDENCE, SEED).encode()).hexdigest()[:20]
            cached = cache / '{0}.csv'.format(signature)
            previous = EXPORT_ROOT / 'plots' / 'ci_cache' / cached.name
            if not cached.is_file() and previous.is_file():
                cached = previous
            if cached.is_file():
                intervals = pd.read_csv(cached)
            else:
                load_path = path
                if row['model'] in source_frame.columns and 'target' in source_frame:
                    fields = {'participant_id': source_frame['participant_id'], 'target': source_frame['target'], 'probability' if task == 'binary' else 'prediction': source_frame[row['model']]}
                    if 'split' in source_frame:
                        fields['split'] = source_frame['split']
                    load_path = cache / '{0}_predictions.csv'.format(signature)
                    pd.DataFrame(fields).to_csv(load_path, index=False)
                frame, detected = load_predictions(load_path, task=task, threshold=0.5, duplicate_policy='error', require_participant_id=True)
                intervals = summarize_model(row['model'], frame, task, BOOTSTRAPS, SEED, CONFIDENCE, regression_strata='auto')
                intervals.to_csv(cached, index=False)
            for _, ci in intervals.iterrows():
                mask = tidy['_row_id'].eq(row_id) & tidy['metric'].eq(metric_name(ci['metric']))
                tidy.loc[mask, ['estimate', 'ci_low', 'ci_high', 'confidence_level', 'ci_status', 'ci_source', 'interval_method', 'n_participants', 'n_bootstrap_requested', 'n_bootstrap_valid']] = [ci['estimate'], ci['ci_low'], ci['ci_high'], ci['confidence_level'], 'participant_bootstrap', original, ci['interval_method'], ci['n_participants'], ci['n_bootstrap_requested'], ci['n_bootstrap_valid']]
            ci_audit.append({**entry, 'status': 'completed', 'prediction_file': original, 'cache': str(cached), 'reason': ''})
        except Exception as error:
            ci_audit.append({**entry, 'status': 'failed', 'prediction_file': original, 'reason': '{0}: {1}'.format(type(error).__name__, error)})
if not tidy.empty:
    tidy['has_95ci'] = np.isclose(pd.to_numeric(tidy['confidence_level'], errors='coerce'), 0.95) & np.isfinite(pd.to_numeric(tidy['ci_low'], errors='coerce')) & np.isfinite(pd.to_numeric(tidy['ci_high'], errors='coerce')) & tidy['ci_low'].le(tidy['ci_high'])
    tidy.to_csv(OUTPUT / 'tidy_metrics.csv', index=False)
    tidy.loc[~tidy['has_95ci']].to_csv(OUTPUT / 'missing_confidence_intervals.csv', index=False)
    display(tidy.groupby(['platform', 'evidence_level', 'resampling_unit', 'has_95ci'], dropna=False).size().rename('metric_rows').reset_index())
pd.DataFrame(ci_audit, columns=sorted(set().union(*(row.keys() for row in ci_audit))) if ci_audit else ['row_id', 'run', 'model', 'status', 'reason']).to_csv(OUTPUT / 'ci_computation_inventory.csv', index=False)
if not saved_ci.empty:
    saved_ci.loc[~saved_ci.index.isin(linked_intervals)].to_csv(OUTPUT / 'saved_ci_unlinked.csv', index=False)

def figure_row_label(row):
    short = row['run'].split('_NEXT_', 1)[-1]
    short = re.sub('_(?:F|M)_' + tag_pattern + '_seed\\d+$', '', short)
    short = re.sub('^(territory_classification|territory_volume|plaque|sensitivity|weight)_', '', short)
    names = {'binary_age': 'Any peripheral plaque\nScan + age', 'binary_age_weight': 'Any peripheral plaque\nScan + age + weight', 'regression_age': 'Peripheral plaque volume\nScan + age', 'regression_age_weight': 'Peripheral plaque volume\nScan + age + weight', 'all_raw': 'Total volume, original target scale', 'all_log': 'Total volume, log1p target', 'all_log_gaussian_mmd': 'Log volume, Gaussian MMD prior', 'all_log_boson_mmd': 'Log volume, Boson MMD prior', 'positive_log': 'Plaque present, log1p target', 'positive_median': 'Plaque present, median-oriented loss', 'hurdle': 'Two-part presence and volume model', 'age_top_log': 'High plaque volume within age bands', 'top5_log': 'Highest 5% measured plaque volume'}
    label = names.get(short, short.replace('_presence', ' plaque presence').replace('_volume', ' plaque volume').replace('_', ' ').capitalize())
    n = scalar(row.get('n_participants', np.nan))
    details = []
    if row['seed']:
        details.append('seed {0}'.format(row['seed']))
    if np.isfinite(n):
        details.append('n={0}'.format(int(n)))
    return label + ('\n' + ', '.join(details) if details else '')
CI_OUTPUT = OUTPUT / 'confidence_intervals'
CI_OUTPUT.mkdir(exist_ok=True)
plot_manifest = []
if not tidy.empty:
    eligible = tidy['evaluation_split'].astype(str).str.lower().isin(['test', 'heldout', 'held_out', 'held-out']) & tidy['resampling_unit'].eq('participant')
    candidates = tidy.loc[eligible & tidy['metric'].isin(PLOT_METRICS)].copy()
    candidates.loc[~candidates['has_95ci']].to_csv(CI_OUTPUT / 'missing_95ci.csv', index=False)
    rows = candidates.loc[candidates['has_95ci']].copy()
    unique_model_metric = ['platform', 'source_root', 'run', 'target', 'cohort', 'gender', 'seed', 'metric', 'evaluation_split']
    rows = rows.sort_values(['_source_priority', 'source_file', 'source_row'], kind='stable').drop_duplicates(unique_model_metric, keep='first')
    rows.to_csv(CI_OUTPUT / 'recent_model_95ci.csv', index=False)
    total_target = rows['target'].isin(['peripheral_total', 'volume_peripheral_total', 'plaque_volume_total_corrected'])
    whole = rows['cohort'].isin(['', 'all', 'whole_cohort', 'all_participants'])
    plans = [('01_total_plaque_mae_95ci', 'Total peripheral plaque volume: whole cohort', 'mae', total_target & whole), ('02_total_plaque_r2_95ci', 'Total peripheral plaque volume: whole cohort', 'r2', total_target & whole), ('03_plaque_subgroups_mae_95ci', 'Total plaque volume: exploratory subgroups', 'mae', total_target & ~whole), ('04_plaque_presence_auc_95ci', 'Plaque presence: recent models', 'auc', rows['plot_family'].isin(['territory_classification', 'weight'])), ('05_territory_volume_mae_95ci', 'Plaque volume by territory: recent models', 'mae', rows['plot_family'].eq('territory_volume'))]
    panel_fields = ['platform', 'source_root', 'cohort', 'gender']
    for stem, title, metric, mask in plans:
        group = rows.loc[mask & rows['metric'].eq(metric)].copy()
        if group.empty:
            continue
        group['display_label'] = group.apply(figure_row_label, axis=1)
        group = group.sort_values(['run', 'model', 'seed', 'source_file'], kind='stable')
        for field in panel_fields:
            group[field] = group[field].replace('', 'unspecified')
        panels = list(group.groupby(panel_fields, dropna=False, sort=True))
        ncols = min(2, len(panels))
        nrows = int(np.ceil(len(panels) / ncols))
        largest = max((len(frame) for _, frame in panels))
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, nrows * max(4.8, 0.7 * largest + 2.2)), squeeze=False)
        for ax, (key, frame) in zip(axes.flat, panels):
            labels = []
            for y, (_, row) in enumerate(frame.iterrows()):
                ax.hlines(y, row['ci_low'], row['ci_high'], color='#0F6F73', lw=2)
                ax.vlines([row['ci_low'], row['ci_high']], y - 0.07, y + 0.07, color='#0F6F73', lw=1.4)
                ax.scatter(row['estimate'], y, color='#0F6F73', s=36, zorder=3)
                labels.append(figure_row_label(row))
            ax.set_yticks(np.arange(len(frame)), labels, fontsize=10)
            ax.set_ylim(largest - 0.5, -0.5)
            platform, root, cohort, gender = key
            short_root = '/'.join(normalized_path(root).split('/')[-2:])
            ax.set_title('{0} | {1} | {2}\n{3}'.format(platform, gender, cohort, short_root), fontsize=10)
            ax.set_xlabel('MAE (reported volume units)' if metric == 'mae' else 'R²' if metric == 'r2' else 'AUC')
            if metric == 'auc':
                ax.set_xlim(max(0, min(0.5, frame['ci_low'].min() - 0.03)), 1)
                ax.axvline(0.5, color='#899499', linestyle='--', lw=0.8)
            if metric == 'r2':
                ax.axvline(0, color='#899499', linestyle='--', lw=0.8)
            ax.grid(axis='x', color='#E4E8E9', lw=0.7)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        fig.suptitle(title + ' — test performance with 95% confidence intervals', fontsize=13)
        fig.text(0.01, 0.005, 'Dots: fitted-model test scores. Bars: 95% participant intervals. Targets/cohorts retain their own participants; no pooling.\nIndividual performance intervals are not paired-effect or training-seed intervals.', fontsize=8)
        fig.tight_layout(rect=(0, 0.045, 1, 0.965))
        group.to_csv(CI_OUTPUT / '{0}.csv'.format(stem), index=False)
        png, pdf = (CI_OUTPUT / '{0}.png'.format(stem), CI_OUTPUT / '{0}.pdf'.format(stem))
        fig.savefig(png, dpi=160, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)
        plot_manifest.append(dict(figure=stem, metric=metric, rows=len(group), panels=len(panels), png=str(png), pdf=str(pdf)))
        display(Image(filename=str(png)))
pd.DataFrame(plot_manifest, columns=['figure', 'metric', 'rows', 'panels', 'png', 'pdf']).to_csv(OUTPUT / 'performance_plot_manifest.csv', index=False)
TOTAL_TARGETS = {'peripheral_total', 'volume_peripheral_total', 'plaque_volume_total_corrected'}
total = tidy.loc[tidy['target'].isin(TOTAL_TARGETS)].copy() if not tidy.empty else pd.DataFrame()
if not total.empty:
    total.to_csv(OUTPUT / 'total_peripheral_volume_metrics.csv', index=False)
    display(total[['platform', 'source_root', 'run', 'model', 'cohort', 'gender', 'seed', 'metric', 'estimate', 'ci_low', 'ci_high', 'ci_status', 'evaluation_split']])
if not tidy.empty:
    neural = tidy['model'].eq('meshvae') | tidy['model'].eq(tidy['run']) | tidy['model'].str.contains('boson', case=False)
    matched = tidy['run'].str.contains('all_log_(?:gaussian_mmd|boson_mmd)', case=False, regex=True) & neural
    boson = tidy.loc[matched | tidy['prior'].str.contains('boson', case=False) & neural | tidy['model'].str.contains('boson', case=False) | tidy['result_scope'].str.startswith('seed_')]
    boson.to_csv(OUTPUT / 'boson_metrics.csv', index=False)
    if not boson.empty:
        display(boson[['platform', 'run', 'model', 'target', 'gender', 'seed', 'result_scope', 'metric', 'estimate', 'resampling_unit', 'has_95ci']])
    paired = tidy.loc[matched & tidy['result_scope'].eq('model_evaluation')].copy()
    if not paired.empty:

        def verified_hashes(value):
            try:
                hashes = json.loads(value)
                return isinstance(hashes, dict) and bool(hashes) and all((str(item).strip() for item in hashes.values()))
            except (TypeError, json.JSONDecodeError):
                return False
        verified = paired['reference_fingerprint'].str.strip().ne('') & paired['reference_split_hashes'].map(verified_hashes)
        paired.loc[~verified].to_csv(OUTPUT / 'boson_pairing_unverified.csv', index=False)
        if not verified.all():
            pass
        paired = paired.loc[verified].copy()
        paired['arm'] = np.where(paired['run'].str.contains('all_log_boson_mmd', case=False), 'boson_mmd', 'gaussian_mmd')
        keys = ['platform', 'source_root', 'target', 'cohort', 'gender', 'seed', 'metric', 'metric_scale', 'reference_fingerprint', 'reference_split_hashes']
        if paired.duplicated(keys + ['arm']).any():
            paired.to_csv(OUTPUT / 'boson_pairing_ambiguous.csv', index=False)
        else:
            comparison = paired.pivot(index=keys, columns='arm', values='estimate').reset_index()
            if {'boson_mmd', 'gaussian_mmd'}.issubset(comparison):
                comparison['boson_minus_gaussian'] = comparison['boson_mmd'] - comparison['gaussian_mmd']
                comparison.to_csv(OUTPUT / 'boson_matched_seed_differences.csv', index=False)
                display(comparison)

def load_plot_metadata(path, job):
    source = read_table(path)
    id_column = resolve_id_column(source, job.get('id_col'))
    age_column = job.get('age_col') or find_column(source, AGE_ALIASES)
    gender_column = job.get('gender_col') or find_column(source, GENDER_ALIASES)
    if age_column not in source or gender_column not in source:
        raise KeyError('Age or sex metadata column is missing.')
    metadata = bmi_plot_data(source, job.get('bmi_col'))
    metadata['participant_id'] = source[id_column].map(clean_participant_id)
    metadata['age'] = numeric(source[age_column])
    metadata['gender'] = normalize_gender(source[gender_column])
    metadata['target'] = resolve_binary_target(source, job['target_col'])
    metadata['metadata_file'] = str(path)
    metadata = metadata.dropna(subset=['participant_id'])
    if metadata['participant_id'].duplicated().any():
        raise ValueError('Duplicate participant IDs in plotting metadata.')
    return (metadata.reset_index(drop=True), source)

def merge_plot_split(metadata, latent, columns, split):
    merged = metadata.merge(latent[['participant_id', *columns]], on='participant_id', how='inner', validate='one_to_one')
    keep = np.isfinite(merged[columns].to_numpy(dtype=float)).all(axis=1) & np.isfinite(merged['age']) & merged['gender'].isin(['F', 'M']) & merged['target'].isin([0, 1])
    result = merged.loc[keep].reset_index(drop=True)
    return (result, dict(split=split, metadata_rows=len(metadata), with_latent_match=len(merged), complete_case_rows=len(result), missing_bmi_retained=int(result['bmi'].isna().sum())))
latent_jobs = list(LATENT_JOBS)
latent_status = []
if not artifacts.empty and {'source_file', 'exported_file'}.issubset(artifacts):
    for _, a in artifacts.loc[artifacts['source_file'].map(normalized_path).str.endswith('/latent_manifest.json')].iterrows():
        try:
            manifest = json.loads(artifact_path(a).read_text(encoding='utf-8'))
            folder = normalized_path(a['source_file']).rsplit('/', 1)[0]
            study_original = folder.split('/study/', 1)[0] + '/study/manifest.json' if '/study/' in folder else normalized_path(a.get('run_directory', '')) + '/study/manifest.json'
            study_path = local_artifact(study_original, a['_export_dir'])
            study = json.loads(study_path.read_text(encoding='utf-8')) if study_path else {}
            territory = study.get('experiment', {}).get('territory')
            target_col = manifest.get('target_col') or ('presence_' + territory if territory else None)
            volume_col = manifest.get('volume_col') or ('volume_' + territory if territory else None)
            wanted = '^{0}_NEXT_{1}_(F|M)_{2}_seed{3}$'.format(TSNE_PLATFORM.upper(), re.escape(TSNE_EXPERIMENT), re.escape(TSNE_RUN_TAG), TSNE_SEED)
            if not re.fullmatch(wanted, manifest['experiment']):
                continue
            latent_jobs.append(dict(name=manifest['experiment'], gender=manifest['gender'], representation_label=manifest.get('representation_label') or '{0}; posterior mean; inputs unverified'.format(manifest['experiment']), train_latent=folder + '/train_latent_embeddings.csv', test_latent=folder + '/test_latent_embeddings.csv', train_tsv=manifest['train_tsv'], test_tsv=manifest['test_tsv'], target_col=target_col, volume_col=volume_col, id_col=manifest.get('id_col'), age_col=manifest.get('age_col'), gender_col=manifest.get('gender_col'), bmi_col=manifest.get('bmi_col'), export_dir=a['_export_dir']))
        except Exception as error:
            latent_status.append(dict(name=a['source_file'], status='bad_manifest', reason=str(error)))
latent_jobs.sort(key=lambda job: (0 if '/MQVAE_new/checkpoints/' in normalized_path(job['train_latent']) else 1, job['name'], str(job['train_latent'])))
seen_jobs = set()
for job in latent_jobs:
    identity = (job['name'], job['gender'], str(job['train_latent']), str(job['test_latent']))
    display_identity = (job['name'], job['gender'])
    if display_identity in seen_jobs or len(seen_jobs) >= MAX_TSNE_FIGURES:
        continue
    seen_jobs.add(display_identity)
    status = {'name': job['name'], 'gender': job['gender'], 'representation': job.get('representation_label', 'Inputs unverified')}
    if not RUN_TSNE:
        latent_status.append({**status, 'status': 'disabled'})
        continue
    resolved = {key: local_artifact(job[key], job.get('export_dir')) for key in ['train_latent', 'test_latent', 'train_tsv', 'test_tsv']}
    absent = [key for key, path in resolved.items() if path is None]
    if absent:
        latent_status.append({**status, 'status': 'missing_inputs', 'reason': ', '.join(absent)})
        continue
    try:
        train_latent, train_cols = load_latent(resolved['train_latent'], 'train', job.get('id_col'))
        test_latent, test_cols = load_latent(resolved['test_latent'], 'test', job.get('id_col'))
        columns = sorted(set(train_cols) & set(test_cols))
        if not columns:
            raise ValueError('No matching latent dimensions.')
        if set(train_latent.participant_id) & set(test_latent.participant_id):
            raise ValueError('Train/test participant overlap.')
        frames, flows, bmi_source_audit = ([], [], [])
        source_columns = read_table(resolved['train_tsv']).columns
        if not job.get('id_col') and 'participant_id' in source_columns:
            job['id_col'] = 'participant_id'
        if not job.get('age_col') and 'study_age' in source_columns:
            job['age_col'] = 'study_age'
        if not job.get('gender_col') and 'study_gender' in source_columns:
            job['gender_col'] = 'study_gender'
        if not job.get('target_col'):
            job['target_col'] = 'presence_peripheral_total' if 'presence_peripheral_total' in source_columns else 'anyplaq'
        for split, latent in [('train', train_latent), ('test', test_latent)]:
            metadata, source = load_plot_metadata(resolved['{0}_tsv'.format(split)], job)
            bmi_source_audit.append(metadata.assign(split=split))
            volume_col = job.get('volume_col') or next((c for c in ['volume_peripheral_total', 'plaque_volume_total_corrected', 'peripheral_total'] if c in source), None)
            if volume_col:
                if volume_col not in source:
                    raise KeyError('Missing measured volume column {0!r}.'.format(volume_col))
                colors = pd.DataFrame({'participant_id': source[resolve_id_column(source, job.get('id_col'))].map(clean_participant_id), 'plaque_volume': numeric(source[volume_col])})
                colors['plaque_volume'] = colors['plaque_volume'].where(np.isfinite(colors['plaque_volume']) & colors['plaque_volume'].ge(0))
                metadata = metadata.merge(colors, on='participant_id', how='left', validate='one_to_one')
            frame, flow = merge_plot_split(metadata, latent, columns, split)
            frame = frame.loc[frame['gender'].eq(job['gender'])].reset_index(drop=True)
            flow['selected_gender_rows'] = len(frame)
            frames.append(frame)
            flows.append(flow)
        train, test = frames
        if len(train) < 4 or len(test) < 3 or len(train) + len(test) < 8:
            raise ValueError('Too few complete participants for t-SNE.')
        fingerprint = hashlib.sha256('|'.join(map(str, identity)).encode() + b''.join((path.read_bytes() for path in resolved.values())) + '{0}|{1}'.format(SEED, PERPLEXITY).encode()).hexdigest()[:12]
        folder = OUTPUT / 'tsne' / (re.sub('[^A-Za-z0-9_.-]+', '_', job['name']) + '_' + job['gender'] + '_' + fingerprint)
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(flows).to_csv(folder / 'cohort_flow.csv', index=False)
        source_audit = pd.concat(bmi_source_audit, ignore_index=True)
        source_audit[[c for c in source_audit if c in ['participant_id', 'split', 'metadata_file'] or c.startswith('bmi')]].to_csv(folder / 'bmi_source_audit.csv', index=False)
        train_values, test_values, info = standardize_latent(train, test, columns, SEED)
        expected = pd.concat([train.assign(split='train'), test.assign(split='test')], ignore_index=True)
        signature = tsne_coordinate_fingerprint(np.concatenate([train_values, test_values]), expected, SEED, PERPLEXITY)
        coordinates = None
        signature_path = folder / 'tsne_coordinate_cache.json'
        if (folder / 'tsne_coordinates.csv').is_file() and signature_path.is_file():
            saved = pd.read_csv(folder / 'tsne_coordinates.csv', dtype={'participant_id': str})
            saved_signature = json.loads(signature_path.read_text(encoding='utf-8'))
            if saved_signature.get('fingerprint') == signature and {'participant_id', 'split', 'tsne_1', 'tsne_2'}.issubset(saved):
                if list(zip(saved.participant_id, saved.split)) == list(zip(expected.participant_id, expected.split)):
                    coordinates = saved[['tsne_1', 'tsne_2']].to_numpy(dtype=float)
        kmeans, scores, k = choose_kmeans(train_values, 2, 8, SEED)
        scores.to_csv(folder / 'cluster_selection.csv', index=False)
        info.update(make_tsne_figure(train_values, test_values, train, test, kmeans.predict(train_values), kmeans.predict(test_values), folder, SEED, PERPLEXITY, job['gender'], job.get('representation_label', 'Inputs unverified'), bmi_limits=BMI_COLOR_LIMITS, coordinates=coordinates))
        (folder / 'analysis_manifest.json').write_text(json.dumps({**job, **info, 'bmi_color_limits': BMI_COLOR_LIMITS, 'note': 'Descriptive embedding; training-fitted scaler/PCA/clusters; source BMI preserved.'}, indent=2, default=str), encoding='utf-8')
        display(Image(filename=str(folder / 'tsne_latent_signal.png')))
        latent_status.append({**status, 'status': 'completed', 'output': str(folder), 'n_train': len(train), 'n_test': len(test), 'volume_panel': 'plaque_volume' in train, 'bmi_audit': str(folder / 'bmi_audit.csv'), 'bmi_source_audit': str(folder / 'bmi_source_audit.csv')})
    except Exception as error:
        latent_status.append({**status, 'status': 'failed', 'reason': '{0}: {1}'.format(type(error).__name__, error)})
pd.DataFrame(latent_status, columns=sorted(set().union(*(row.keys() for row in latent_status))) if latent_status else ['name', 'status', 'reason']).to_csv(OUTPUT / 'latent_input_inventory.csv', index=False)
if latent_status:
    display(pd.DataFrame(latent_status))
display(pd.DataFrame({'output': ['confidence_intervals/', 'tsne/', 'recent_run_selection.csv', 'tidy_metrics.csv', 'performance_plot_manifest.csv', 'missing_confidence_intervals.csv', 'ci_computation_inventory.csv', 'latent_input_inventory.csv'], 'folder': [str(OUTPUT)] * 8}))
(OUTPUT / 'notebook_manifest.json').write_text(json.dumps({'exports': [str(path) for path in EXPORT_DIRS], 'bootstrap_replicates': BOOTSTRAPS, 'confidence': CONFIDENCE, 'seed': SEED, 'confidence_interval_figures': len(plot_manifest), 'recent_run_tags': RECENT_RUN_TAGS, 'suite_mode': RECENT_SUITE_MODE, 'max_tsne_figures': MAX_TSNE_FIGURES, 'latent_jobs': len(latent_jobs), 'bmi_color_limits': BMI_COLOR_LIMITS, 'plot_version': '20261014', 'note': 'Original provenance preserved; missing intervals explicit; no training.'}, indent=2), encoding='utf-8')
