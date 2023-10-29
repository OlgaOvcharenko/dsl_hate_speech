#!/bin/bash

mkdir -p models

declare -a models=("Hate-speech-CNERG/dehatebert-mono-german"  "statworx/bert-base-german-cased-finetuned-swiss" "deepset/bert-base-german-cased-hatespeech-GermEval18Coarse" "german-nlp-group/electra-base-german-uncased")


for value in "${models[@]}"
do
  sbatch run.sh $value
  # bash run.sh $value
done
