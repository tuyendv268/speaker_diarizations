from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import reclustering
import os

ROOT = os.getcwd()
data_dir = os.path.join(ROOT,'conf')
os.makedirs(data_dir, exist_ok=True)
# MODEL_CONFIG = os.path.join(data_dir,'diar_infer_meeting.yaml')
MODEL_CONFIG = os.path.join(data_dir,'diar_infer.yaml')

config = OmegaConf.load(MODEL_CONFIG)
# /home/tuyendv/projects/speaker-diarization/datas/nemo_datas/valid/msdd_data.json
config.diarizer.manifest_filepath = '/home/tuyendv/projects/speaker-diarization/datas/nemo_datas/valid/msdd_data.json'
# config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5] 
# config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1] 
# config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1] 
config.diarizer.clustering.parameters.oracle_num_speakers = False

pretrained_vad = 'vad_multilingual_marblenet'
pretrained_speaker_model = 'titanet_large'

config.num_workers = 1 # Workaround for multiprocessing hanging with ipython issue 

output_dir = os.path.join(ROOT, 'outputs')
config.diarizer.out_dir = output_dir #Directory to store intermediate files and prediction outputs

config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
# diar_msdd_telephonic
config.diarizer.msdd_model.model_path = "/home/tuyendv/projects/speaker-diarization/nemo_experiments/MultiscaleDiarDecoder/2022-11-18_10-04-02/checkpoints/MultiscaleDiarDecoder--val_loss=9.8792-epoch=8.ckpt"
config.diarizer.oracle_vad = False # compute VAD provided with model_path to vad config
config.diarizer.clustering.parameters.oracle_num_speakers=False

# Here, we use our in-house pretrained NeMo VAD model
config.diarizer.vad.model_path = pretrained_vad
# config.diarizer.vad.parameters.onset = 0.8
# config.diarizer.vad.parameters.offset = 0.6
# config.diarizer.vad.parameters.pad_offset = -0.05
print(OmegaConf.to_yaml(config))


print(f"VAD params:{OmegaConf.to_yaml(config.diarizer.vad.parameters)}")

# config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic'
config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0] # Evaluate with T=0.7 and T=1.0
system_vad_msdd_model = NeuralDiarizer(cfg=config)
import time

t = time.time()
print("------------- diarize --------------")
system_vad_msdd_model.diarize()
# reclustering.convert_rttm_to_segments(abs_rttm_path, abs_wav_path)
print("------------- global clustering --------------")
# global_clustering()
t1 = time.time()

print(t1-t)