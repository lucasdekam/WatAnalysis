# WatAnalysis

This is a package for (parallel) analysis of water structures/dynamics at **metal/water interfaces**.

## Installation
To install this fork of WatAnalysis:
```bash
git clone https://github.com/lucasdekam/WatAnalysis.git
cd WatAnalysis
pip install .
```

## User Guide
Command-line interface:

```
watanalysis --help
```

For example, for a typical gold-water interface:

```
watanalysis --pattern "./oc25_w_charge/pos_traj.xtc" --nprocs 32 --interface 0 4.5
```

You can also build custom scripts based on WatAnalysis; the CLI script `WatAnalysis/cli/cli.py` serves as an usage example.
