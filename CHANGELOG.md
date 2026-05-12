# CHANGELOG

## 20260512 v0.3
1. Remove undesired contact penalty.
2. Add `rsl_rl 5.3.0` locally to add custom modifications, add git commit in git diff storing.
3. Add new README.md

## 20260511 v0.2
1. Upload dailylife dataset to huggingface https://huggingface.co/datasets/wty-yy/beyondmimic_dailylife_dataset
2. Add python start training script [start_beyondmimic.py](scripts/start/start_beyondmimic.py) for multi-motion model training, each motion class folder will be trained by a single model.
3. Change default mimic `max_iterations=50000`

## 20260501 v0.1
1. Add multi motion tracking support for `csv_to_npz.py`, `train.py`, `play.py`

We found a bug for original beyondmimic adaptive sampling, there are three types terminal need to resample:
1. motion finished naturally: no failure penalty
2. timeout: no failure penalty
3. bad-tracking error causes termination: update EMA and failed bin count

The original implementation used the environment termination signal directly for adaptive sampling, so it did not distinguish true bad-tracking failures from other reset/termination cases.
