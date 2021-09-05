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
- [Cite](#cite)

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
To enable apex mixed precision, and to download and prepare the dataset, 
please see the instructions [here](https://githubcom/yuvalkirstain/s2e-coref#download-the-official-evaluation-script).

## Evaluation
Download our trained model:
####TODO: curl link
 ```
export MODEL_DIR=<model_dir>
curl -L  > temp_model.zip
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
            --lr 1e-4 \
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
            --lr 1e-4 \
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
