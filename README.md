# Generative Coreference Resolution

This repository contains the code implementation from our project 
["Generative Coreference Resolution"](https://drive.google.com/file/d/1bBudCr0Ndlaq4PDmq2aaTmLmhH2V3L4w/view?usp=sharing).

Our code is based upon the work of the paper ["Coreference Resolution 
without Span Representations"](https://www.semanticscholar.org/paper/Coreference-Resolution-without-Span-Representations-Kirstain-Ram/3029263ca51e6c2907f9f99277083cf6afb1adb7)
[[1]](#1).


- [Set up](#set-up)
  * [Requirements](#requirements)
  * [Download the official evaluation script](#download-the-official-evaluation-script)
  * [Prepare the dataset](#prepare-the-dataset)
- [Evaluation](#evaluation)
- [Training](#training)

## Set up

#### Requirements
Clone "Coreference Resolution 
without Span Representations"'s [repository](https://github.com/yuvalkirstain/s2e-coref):
```
git clone https://github.com/yuvalkirstain/s2e-coref.git
```
Copy the required changes for running:
```
cp -r src/* s2e-coref/
```
Install the requirements:
```
cd s2e-coref
pip install -r requirements.txt
```
#### Download the official evaluation script
Run (from inside the repo):
 
```
git clone https://github.com/conll/reference-coreference-scorers.git
```

#### Prepare the dataset

This repo assumes access to the [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) corpus.
Convert the original dataset into jsonlines format using:
```
export DATA_DIR=<data_dir>
python minimize.py $DATA_DIR
``` 
Credit: This script was taken from the [e2e-coref](https://github.com/kentonl/e2e-coref/) repo.

## Evaluation
Download our trained model:
 ```
export MODEL_DIR=<model_dir>
gdown --id 1uPzu-wAnMoO84tK_urRLxO7zN6eRQ2fy --output temp_model.zip
unzip temp_model.zip -d $MODEL_DIR
rm -rf temp_model.zip
```

and run:
```
export OUTPUT_DIR=<output_dir>
export CACHE_DIR=<cache_dir>
export MODEL_DIR=<model_dir>
export DATA_DIR=<data_dir>
export SPLIT_FOR_EVAL=<dev or test>
python run_config.py \
            --model_type t5-base \
            --split_for_eval test \
            --epochs 1 \
            --model_name_or_path $MODEL_DIR \
            --sent_num 10 \
            --step_num 10
```


## Training
Train a coreference model using the run_config.py configuration:
```
export OUTPUT_DIR=<output_dir>
export CACHE_DIR=<cache_dir>
export MODEL_DIR=<model_dir>
export DATA_DIR=<data_dir>
export SPLIT_FOR_EVAL=<dev or test>
python run_config.py \
            --model_type t5-base \
            --split_for_eval test \
            --epochs <num of epochs> \
            --model_name_or_path $MODEL_DIR \
            --sent_num 10 \
            --step_num 10 \
            --do_train
```
For changes of more parameters run:
```
python run_config.py \
           run_config.py -h
```
and run accordingly


To evaluate your trained model on test go [here](#evaluation).

## References
<a id="1">[1]</a> 
[Coreference Resolution without Span Representations](arXiv:2101.00434), 2021, Kirstain et 
al.
