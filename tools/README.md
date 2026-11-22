# Installer builders

These helpers package the organized Python entry points into standalone
Computerome setup scripts. Run them from the repository root:

```bash
python tools/measurement_baseline_delivery/build_setup.py
python tools/measurement_pointcloud_delivery/build_installer.py
python tools/measurement_expansion_delivery/build_setup.py
```

The generated scripts are written to `scripts/computerome/`. Regenerate an installer
after changing the corresponding Python entry point. Each installer includes the
measurement scripts, package helpers, and configurations it needs, and installs
them into the same organized folder hierarchy. Existing files are backed up.

The expansion workflow also has a synthetic-data smoke check:

```bash
python tools/measurement_expansion_delivery/smoke.py
```

It requires the measurement workflow's dependencies and records its result beside
the smoke-check script.
