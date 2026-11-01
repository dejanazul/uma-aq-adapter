# UMA-AQ adapter

Code for

> D. A. Ultamar. *Adapting UMA to spin-polarized AQCat25 without catastrophic forgetting with task-gated LoRA.*
> ChemRxiv (2026). [doi:10.26434/chemrxiv.15010987](https://doi.org/10.26434/chemrxiv.15010987)

UMA-AQ adapter adds a new task, `aqcat25`, to the universal interatomic potential UMA-S-1p2 and trains only parts
that belong to that task: its dataset embedding, its expert slot in the energy head, and rank-8 LoRA adapters on the
backbone MoLE layers that are switched on only for `aqcat25` systems (188,672 parameters in total). Every other UMA
task keeps the released weights and gives the same predictions as before. On the AQCat25 OOD-Both splits the
adsorption-energy MAE falls from 0.193 to 0.119 eV (test) and from 0.173 to 0.116 eV (val).

| Record | Content | DOI |
|---|---|---|
| Record 1 (this repository) | Code, MIT license | [10.5281/zenodo.23860497](https://doi.org/10.5281/zenodo.23860497) |
| Record 2a | Predictions, logs, data splits, analysis plans; CC BY-NC-SA 4.0 | [10.5281/zenodo.23860707](https://doi.org/10.5281/zenodo.23860707) |
| Record 2b | Trained adapter weights; FAIR Chemistry License v1, available on request | [10.5281/zenodo.23860868](https://doi.org/10.5281/zenodo.23860868) |

## Repository layout

| Path | Content |
|---|---|
| `colab/` | Scripts that run on a GPU VM (written for Google Colab, copied flat to `/content`): data selection and conversion, adding the `aqcat25` task (`adapt_surgery.py`), element-reference refit, training (`adapt_train.yaml`, `adapt_launch.py`, `adapt_lora.py`, `adapt_freeze.py`), checks, final evaluation (`final_eval.py`), adapter export and loading (`adapter_io.py`), and the two baselines (zero-shot UMA, EV2-FiLM). |
| `analysis/zeroshot/` | AQCat25 frame extraction and the zero-shot UMA analysis. |
| `analysis/failure_analysis/` | Zero-shot error analysis; defines the magnetization strata S0–S3 (Sprint 4). |
| `analysis/ev2_film/` | EV2-in+midFiLM baseline. |
| `analysis/final/` | Final evaluation: gates, P1/P2/G tests, per-stratum and original-task results. |
| `analysis/paper/` | `make_tables.py` and `make_figures.py`, which rebuild the paper's tables and figures. |
| `analysis/record2/` | Building Record 2a, removing and restoring AQCat25 labels, downloading AQCat25. |

Analysis scripts are run from the repository root and read their inputs from `results/` and `data/`, which are not
part of this repository (see below).

## Install

Python 3.11–3.13.

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `fairchem-core==2.23.0` (the version used throughout) and `fairchem-data-oc==1.0.2`; the
other packages are left unpinned. The exact environment of the final evaluation is in Record 2a,
`results/adapter/final/final_pip_freeze.txt`. The EV2-FiLM baseline needs its own environment (fairchem-core 1.10.0
with SandboxAQ's patch, torch 2.4.0), which `colab/film_setup.sh` creates.

You also need access to two gated resources, under their own licenses:

- **UMA-S-1p2**: Hugging Face `facebook/UMA`, FAIR Chemistry License v1. Needed for training, evaluation and the
  adapter, not for rebuilding the tables and figures.
- **AQCat25**: Hugging Face `SandboxAQ/aqcat25-dataset`, CC BY-NC-SA 4.0, non-commercial use only. We use commit
  `2ec3a36c138ef0150ac9b1d783a7acf0aadfeda9`.

Set `HF_TOKEN` to a Hugging Face token of an account that has been granted access.

## AQCat25 labels: joining them from your own copy

We do not redistribute AQCat25 DFT labels. Record 2a contains the model predictions with every AQCat25 label and
energy error set to `null`. `analysis/record2/labels.py restore` puts them back from your own AQCat25 copy:

1. `analysis/record2/fetch_aqcat25.py` downloads the AQCat25 parquet metadata and structure archives at commit
   `2ec3a36c` (test/val OOD-Both and their slabs, plus `train_id` and `id_slabs` metadata if passed as arguments) and
   the public OC20 mapping file.
2. `analysis/zeroshot/uma_zeroshot_prep.py` writes the evaluated frames, with their DFT energies, to
   `data/zeroshot/uma_zeroshot_frames{,_val}.extxyz`.
3. `labels.py restore` reads those energies, checks a SHA-256 of all (key, energy) pairs against the value of the
   release (`e2661b68…`), and refuses if they differ (wrong commit or a modified file). It then fills `dft_total_eV`
   and `total_err_eV` (= prediction − DFT) in the seven result files. The internal-validation rows of the loading
   gate are filled from the `train_id` / `id_slabs` parquet files through `results/adapter/aq_val_frames.csv`.

The restored files are byte-identical to the ones used for the paper.

## Reproducing the tables and figures (CPU)

```bash
# 1. put Record 2a's results/ folder in the repository root
cp -r /path/to/record2a/results .

# 2. AQCat25 at commit 2ec3a36c and the OC20 mapping
python -I analysis/record2/fetch_aqcat25.py train_id id_slabs

# 3. AQCat25 frames and metadata
python -I analysis/zeroshot/aqcat25_structures.py test_ood_both
python -I analysis/zeroshot/aqcat25_structures.py val_ood_both
python -I analysis/zeroshot/uma_zeroshot_prep.py 10 test
python -I analysis/zeroshot/uma_zeroshot_prep.py 10 val

# 4. labels
python -I analysis/record2/labels.py restore

# 5. analysis chain, in this order
python -I analysis/zeroshot/uma_zeroshot_analyze.py
python -I analysis/failure_analysis/sprint2_analyze.py
python -I analysis/failure_analysis/sprint3_analyze.py
python -I analysis/failure_analysis/sprint4_analyze.py data/oc20_data_mapping.pkl
python -I analysis/ev2_film/stage_m_level_a.py
python -I analysis/ev2_film/stage_m_film_analyze.py
python -I analysis/final/final_gate.py gate
python -I analysis/final/final_gate.py gateh
python -I analysis/final/final_analyze.py        # P1, P2, G, strata, original tasks
python -I analysis/paper/make_figures.py         # results/figures/
python -I analysis/paper/make_tables.py          # Table 1 and the EV2-FiLM tables
```

Each analysis script prints its tables as Markdown and writes its outputs next to its inputs under `results/`. The
paper's main tests are in `results/adapter/final/final_summary.json`; per-relaxation results are in
`results/adapter/final/aq_final_relaxations.csv`.

This sequence was run on 2026-11-01 in a fresh environment, starting only from Record 2a and AQCat25 downloaded from
Hugging Face. All 140 result files and the console output of all 11 analysis scripts were byte-identical to the
release.

## Using the trained adapter

The adapter weights are in Record 2b (restricted access; request access on Zenodo, non-commercial use only).

```bash
python colab/adapt_surgery.py --in uma-s-1p2.pt --out uma-s-1p2-aqcat25.pt \
    --elem-refs aqcat25_element_references.json
python colab/adapter_io.py uma-s-1p2-aqcat25.pt uma_aq_adapter_ema.pt structures.extxyz --device cuda
```

```python
from adapter_io import load_unit             # colab/adapter_io.py
from fairchem.core import FAIRChemCalculator

unit = load_unit('uma-s-1p2-aqcat25.pt', 'uma_aq_adapter_ema.pt', device='cuda')
atoms.calc = FAIRChemCalculator(unit, task_name='aqcat25')
```

The model loaded this way is bit-identical to the model evaluated in the paper. Use `merge_mole=False` (the default
of `load_unit`): merged MoLE layers bypass the adapters. The other task names (`oc20`, `omat`, `omol`, `oc22`,
`oc25`, `odac`, `omc`) give the released UMA-S-1p2 behaviour.

## Re-running training and evaluation (GPU)

The scripts in `colab/` were run on Google Colab and assume its layout: scripts copied to `/content`, a Hugging Face
token in `/content/hf_token`, and data and checkpoints in a Google Drive folder `aqcat25-lab-data/` mounted at
`/content/drive/MyDrive`. Adjust those paths for another machine.

| Step | Scripts | What it does |
|---|---|---|
| Data | `adapt_prep.sh` → `adapt_select.py`, `adapt_data.py` | Selects relaxations from `train_id`/`id_slabs` (paper filter, strata, 129 held-out bulks, seed 0), converts them to ASE-LMDB, drops organic slabs; converts the OC20 S2EF 2M replay data. |
| New task and gates | `adapt_gates.sh` → `adapt_surgery.py`, `adapt_gate_s1.py` | Adds the `aqcat25` task to `uma-s-1p2.pt` (gate S0: every other task unchanged) and checks the replay label convention (gate S1). |
| Element references | `adapt_refit_refs.py predict … / fit …`, `adapt_offset_check.py` | Refits the `aqcat25` element references on training frames (checkpoint r2). |
| Training | `adapt_gpu_setup.sh`, `adapt_launch.py`, `adapt_train.yaml`, `adapt_train_job.sh`, `adapt_resume.sh` | Launches fairchem fine-tuning. `l2_chain.sh` is the final run (task-gated LoRA rank 8, 40,000 steps); `u1_chain.sh`, `o1_chain.sh`, `l1_chain.sh` and `adapt_launch.py B …` are the variants of Table 1. |
| Frozen-weight check | `adapt_verify_frozen.py` | Verifies that no shared tensor changed (gate V1). |
| Final evaluation | `final_chain.sh` → `final_eval.py` (steps `gate`, `aq`, `aqz`, `oc20`, `forget`), `final_oc22.py`, `final_diag.py`; `forget_data*.py` downloads the original-task validation sets | Assembles the final model (released EMA backbone + trained `aqcat25` parts) and evaluates AQCat25, OC20 and all original UMA tasks. |
| Adapter export | `adapter_io.py` | Writes the trained parts to a small file (Record 2b). |
| Zero-shot baseline | `uma_zs_setup.sh`, `uma_zs_run.py` → `uma_zs_eval.py` (test); `f2_run.py` → `f2_eval.py` (val) | UMA-S-1p2 zero-shot predictions. |
| EV2-FiLM baseline | `film_setup.sh`, `film_run.py` → `film_eval.py` | SandboxAQ's EV2-FiLM checkpoints on the same frames. |

The analysis plans written for these steps, with their version history, are in Record 2a, `plans/`.

## License

The code in this repository is released under the MIT License (`LICENSE`). It does not contain UMA weights or
AQCat25 data. Using it requires UMA-S-1p2 (FAIR Chemistry License v1) and AQCat25 (CC BY-NC-SA 4.0, non-commercial),
which you obtain under their own terms, and models trained with it from those inputs follow those licenses, not MIT.
`colab/adapt_surgery.py` follows the pattern of fairchem's `uma_1p2p1_surgery.py` (MIT, Copyright (c) Meta, Inc. and
its affiliates).

## Citation

Please cite the paper and this record (`CITATION.cff`):

- D. A. Ultamar. Adapting UMA to spin-polarized AQCat25 without catastrophic forgetting with task-gated LoRA.
  ChemRxiv (2026). doi:10.26434/chemrxiv.15010987
- D. A. Ultamar. UMA-AQ adapter: code. Zenodo (2026). doi:10.5281/zenodo.23860497

and the resources it builds on:

- O. Allam et al. AQCat25: Unlocking spin-aware, high-fidelity machine learning potentials for heterogeneous
  catalysis. *npj Comput. Mater.* **12**, 226 (2026). doi:10.1038/s41524-026-02099-6
- B. M. Wood et al. UMA: A family of universal models for atoms. *NeurIPS* (2025). arXiv:2506.23971
