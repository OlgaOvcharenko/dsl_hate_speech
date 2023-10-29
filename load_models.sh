#!/bin/bash

username=oovcharenko
path_on_euler=/cluster/scratch/oovcharenko/dsl_hate_speech

mkdir -p models

declare -a models=("Hate-speech-CNERG/dehatebert-mono-german"  "statworx/bert-base-german-cased-finetuned-swiss" "deepset/bert-base-german-cased-hatespeech-GermEval18Coarse" "german-nlp-group/electra-base-german-uncased")

python3 load_models.py 

scp -r models $username@euler.ethz.ch:$path_on_euler
