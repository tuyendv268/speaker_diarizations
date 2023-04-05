# speaker-diarization


## Getting started

Install environment similar to [https://github.com/NVIDIA/NeMo]

# Create and Augment data

## 1. Simulate data for speaker diarization
```
# edit the following parameter in [simulate_data_for_speaker_diarization.py]

# num_process = 40 --- num process for simulate data in parallel
# num_sample_per_process = 16 --- num iterations for simulate data per process
# inp_path="datas/preprocessed_datas/train" --- path to input data
# out_path="datas/simulated_datas/train" --- path to output data

# simulation_config.yaml : config for simulated data
# run the following command for simulated data
!python simulate_data_for_speaker_diarization.py

```
## 2. Prepare data for training nemo speaker diarization

```
# Edit path to simulated dataset (train, test, valid) in [prepare_nemo_data_format_for_training.py]
# Run following command to prepare data
!python prepare_nemo_data_format_for_training.py

```

## 3. Augment speaker diarization data (white noise - gaussian noise)

```
# Edit path in [data_augmentation/agument_config.ini]
# Run following command to augment data
!python data_augmentation/main.py

```
# Training

```
# Edit the following param
config.model.train_ds.manifest_filepath = 'path_to_training_data msdd_data.50step.json'
config.model.validation_ds.manifest_filepath = 'path_to_valid_data msdd_data.50step.json'
config.model.test_ds.manifest_filepath = 'path_to_testing_data msdd_data.50step.json

# Run following command to start training
bash train_speaker_diarization.sh

```

# Inference

## 1. Inference with non global clustering
```
cd nemo
# edit mp3_path in [nemo/infer_non_reclustering.py] file
# run
!python infer_non_reclustering.py
# result save to outputs folder
```
## 2. Inference with non global clustering
```
cd nemo
# edit mp3_path in [nemo/inference.py] file
# run
!python inference.py
# result save to outputs folder
```
