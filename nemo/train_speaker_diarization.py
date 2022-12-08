import pytorch_lightning as pl
from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf
import os

NEMO_ROOT = os.getcwd()
MODEL_CONFIG = os.path.join(NEMO_ROOT,'conf/msdd_5scl_15_05_50Povl_256x3x32x2.yaml')
config = OmegaConf.load(MODEL_CONFIG)

config.model.train_ds.manifest_filepath = '/home/tuyendv/projects/vnd-nlp-speaker-diarization/datas/nemo_datas/train/msdd_data.50step.json'
config.model.validation_ds.manifest_filepath = '/home/tuyendv/projects/vnd-nlp-speaker-diarization/datas/nemo_datas/valid/msdd_data.50step.json'
config.model.test_ds.manifest_filepath = '/home/tuyendv/projects/vnd-nlp-speaker-diarization/datas/nemo_datas/test/msdd_data.50step.json'

config.batch_size=12
config.model.emb_batch_size=0
config.model.train_ds.emb_dir="/home/tuyendv/projects/vnd-nlp-speaker-diarization/datas/embedding/train" 
config.model.validation_ds.emb_dir="/home/tuyendv/projects/vnd-nlp-speaker-diarization/datas/embedding/valid" 
config.model.test_ds.emb_dir="/home/tuyendv/projects/vnd-nlp-speaker-diarization/datas/embedding/test" 
config.model.diarizer.speaker_embeddings.model_path="titanet_large"
config.trainer.max_epochs = 20
config.trainer.strategy = None


trainer = pl.Trainer(**config.trainer)
exp_manager(trainer, config.get("exp_manager", None))
msdd_model = EncDecDiarLabelModel(cfg=config.model, trainer=trainer)

trainer.fit(msdd_model)
# msdd_model.save_to("msdd.nemo")