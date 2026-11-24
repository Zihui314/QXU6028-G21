# QXU6028 Polymer Devices · Group G21

The G21 regular-data pipeline reproduces the L1 carrier-counting calculation, L2 model selection and Fermi-level trajectory, and L3 Hall-conductivity reconstruction. The root-level `results.json` contains the machine-readable output.

## Reproduce the analysis

Download [G21_reproducibility.zip](G21_reproducibility.zip), extract it to a new directory, and run these commands from that directory.

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

The archive includes the input data and SHA-256 manifest, pipeline code, numerical tables and figures. The script checks input integrity and regenerates the L1–L3 outputs. The `pipeline_commit` value records the original analysis revision.
