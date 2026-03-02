# Generalization of RLVR Using Causal Reasoning as a Testbed
This repository contains the code and data for the paper: [Generalization of RLVR Using Causal Reasoning as a Testbed](https://arxiv.org/abs/2512.20760).

## Overview

This codebase implements synthetic data generation, RLVR training (based on VeRL), LLM-based errror-analysis of reasoning traces, and visualization of results.

It includes:

- Code for sampling causal queries on graphical models and for computing their ground truth answer.
- Scripts for training LLMs with RLVR/SFT and for running inference.
- Tools for reporting experiment results using tables and plots.
- Tools for error-analysis of reasoning traces using o4-mini.


## Table of Contents

1. [Installation](#installation)
2. [RLCausal Synthetic Data Generation](#rlcausal-synthetic-data-generation)
    - Part 1: [Raw Data Generation](#raw-data-generation)
    - Part 2: [Conversion to VeRL](#conversion-to-verl)
3. [Training and Generation](#training-and-generation)
    - Part 1: [RLVR Training and Generation](#rlvr-training-and-generation)
    - Part 2: [SFT Training and Generation](#rlvr-training-and-generation)
4. [Evaluation](#evaluation)
    - Part 1: [Metrics and Visualization](#metrics-and-visualization)
    - Part 2: [Reasoning Trace Error Analysis](#reasoning-trace-error-analysis)
5. [Artifacts](#artifacts)
7. [Citation](#citation)

## Installation
Run `bash install_dependencies.sh` under project root. It will create a new conda environment called `rlcausal` with the required dependencies, including VeRL. Also run `bash install_base_models.sh` to download the Qwen models if you wish to perform training later.

## RLCausal Synthetic Data Generation
Run `bash generate_data.sh` under project root.

### Raw Data Generation
Set `GENERATE_RAW=1` in `generate_data.sh` to generate / re-generate raw data. The generated data are saved as parquet files under `data/rl` and `data/sft`. Python code and config for the generating process live under `data_generation`.

### Conversion to VeRL
Set `GENERATE_VERL=1` in `generate_data.sh` to generate / re-generate verl format data. It assumes you have already generated the raw data. The generated data are saved as parquet files under `data/verl`.

## Training and Generation
### RLVR Training and Generation
For training run the following command:
```bash
bash training/rlvr/train_rlvr.sh \\
  --level {interventional/observational/counterfactual} \\
  --model_path Qwen/Qwen2.5-{3B/7B/32B}-Instruct
```
Checkpoints by default will be saved every 10 steps with a maximum of 2 kept checkpoints to `checkpoints/rlcausal-rlvr/...`.

For generation run the following command:
```bash
bash inference/gen.sh \\
    --ckpt_model_path path_to_checkpoint
```
For example, if you train rlvr 7B for 10 steps, your checkpoint would be saved to
`checkpoints/rlcausal-rlvr/exp_short_prompt_rlvr_interventional_n10v2_model_Qwen2_5_7B_Instruct_bs8_lr1e-6_roll32_p2048_r4096_ts8k_rwdscore_strict_rt1.0_fltmean_correctness_fltmin-0.01_fltmax1.01/global_step_10/`.

### SFT Training and Generation
To be added.

## Evaluation
To be added.

## Artifacts

Raw and VeRL format data for RLVR and SFT are included under `data`.

More artifacts to be added.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{lu2025generalizationrlvrusingcausal,
      title={Generalization of RLVR Using Causal Reasoning as a Testbed}, 
      author={Brian Lu and Hongyu Zhao and Shuo Sun and Hao Peng and Rui Ding and Hongyuan Mei},
      year={2025},
      eprint={2512.20760},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.20760}, 
}
```