# Final no-leak diagnostic MeshVAE patch

Copy these files over your refactored project:

```bash
cd /data/projects/3D_scans/ML/MQVAE
mkdir -p backup_before_no_leak_$(date +%Y%m%d_%H%M%S)
cp meshvae/options.py meshvae/conditioning.py meshvae/networks.py meshvae/diagnostics.py backup_before_no_leak_*/
unzip -o /path/to/meshvae_final_no_leak_changed_files.zip -d .
```

This patch changes the refactored code, not the thin wrapper `train_mesh_vae.py`.

Changed behavior:

- New option: `--encoder_cond_mode {none,demo}`.
- Use `--encoder_cond_mode demo` for the final diagnostic model.
- Do not use `--encoder_use_cond`; it now raises a safety error.
- With `demo`, the encoder receives only `[gender_M, gender_F, age...]`.
- The decoder and conditional prior still receive the full condition vector `[gender_M, gender_F, age..., athero...]`.
- Atherosclerosis labels never enter the encoder.

For age bins `18 40 50 55 65 200` and `--cond_athero_mode 4bit`:

- encoder condition dim = 7 = 2 gender + 5 age
- decoder/prior condition dim = 11 = 2 gender + 5 age + 4 athero
- atherosclerosis offset = 7
