# CHANGELOG

## 20260501 v0.1
1. Add multi motion tracking support for `csv_to_npz.py`, `train.py`, `play.py`

We found a bug for original beyondmimic adaptive sampling, there are three types terminal need to resample:
1. motion finished naturally: no failure penalty
2. timeout: no failure penalty
3. bad-tracking error causes termination: update EMA and failed bin count

The original implementation used the environment termination signal directly for adaptive sampling, so it did not distinguish true bad-tracking failures from other reset/termination cases.
