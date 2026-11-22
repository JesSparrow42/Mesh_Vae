# MeshVAE refactor with conditional age + adversarial age loss

Run the commands in this guide from the repository root.

This folder keeps the old command names (`train_mesh_vae.py`, `mesh_vae.py`, `mesh_geom_pipeline.py`, `diagnose_mesh_vae.py`) but moves most helper code into the `meshvae/` package.

## Main changes

- `meshvae/boson.py`: all boson sampler and boson prior code.
- `meshvae/geometry.py`: mesh loading, cleaning, simplification, tensor conversion, point-cloud sampling, and redetailing.
- `meshvae/conditioning.py`: metadata parsing, age/gender vectors, condition-vector layout helpers, and `cond_stats.json` writing.
- `meshvae/datasets.py`: raw/preprocessed datasets and collate function.
- `meshvae/networks.py`: encoder, decoder, and MeshVAE network.
- `meshvae/adversarial.py`: gradient reversal layer used by the age adversary.
- `meshvae/vi.py`: ELBO, auxiliary losses, and adversarial age loss.
- `meshvae/model.py`: training wrapper, optimizer, scheduler, checkpoint I/O.
- `meshvae/training.py`: train/test/sample/baseline phases.
- `meshvae/age_bins.py`: age-bin search helpers.
- `meshvae/diagnostics.py`: diagnostic scoring and thresholding pipeline.

## Age is back in the condition vector

The condition vector is now consistently laid out as:

```text
[gender_M, gender_F, age_0, age_1, ..., age_K, optional athero bits, optional plaque volumes]
```

A bug in the old `build_age_gender_lookup()` path was fixed: gender is now actually written into `cvec[0]` / `cvec[1]`, and atherosclerosis/plaque offsets are no longer hard-coded to `6`; they use `2 + number_of_age_bins`.

## Adversarial age loss

Enable it with:

```bash
python train_mesh_vae.py   --phase train   --use_cond   --encoder_use_cond   --use_cond_prior   --adv_predict_age   --adv_age_weight 0.2   --adv_age_grl_lambda 1.0   ...your existing args...
```

The decoder still receives age as a normal conditional VAE input, but the latent `mu(X)` is passed through a gradient-reversal age classifier. The classifier tries to predict the age bin, while the encoder receives the opposite gradient and is discouraged from storing age in the latent representation.

Monitor these diagnostics:

- `age_adv_loss`
- `age_adv_acc`

For 4 age bins, chance accuracy is about `0.25`. If `age_adv_acc` is very high, age is still easy to read from the latent; increase `--adv_age_weight` gradually.

## Recommended first run

```bash
python train_mesh_vae.py   --dataroot ./data   --name REACT_vae_age_adv   --phase train   --use_preprocessed   --use_pointcloud   --use_cond   --encoder_use_cond   --use_cond_prior   --cond_tsv cohort.tsv   --cond_athero   --cond_athero_mode 4bit   --adv_predict_age   --adv_age_weight 0.2   --adv_age_grl_lambda 1.0   --ncf 64 128 256   --latent_dim 64   --ninput_edges 10000
```

If reconstruction gets worse, reduce `--adv_age_weight` to `0.05` or turn off `--encoder_use_cond` while keeping decoder conditioning.
