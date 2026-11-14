#!/usr/bin/env bash
# Reproduces every number in evaluation/results/ (CPU: ~5 h, mostly Phi-2).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
$PY make_splits.py
$PY qa_validation.py
$PY ner_checks.py
$PY retrieval_eval.py
$PY gen_eval.py --model t5 --epochs 0 --tag _zeroshot
$PY gen_eval.py --model t5 --epochs 3
$PY gen_eval.py --model phi2 --epochs 1 --bs 4 --no_val --settings gold,top3
