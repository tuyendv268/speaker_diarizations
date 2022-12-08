from glob import glob
from tqdm import tqdm
import logging
import os

"""
  preparing dataset for training nemo speaker diarization
"""

def create_rttmlist_and_wavlist(input_path, output_path):
    # clean output dir
    os.system(f'rm -r {output_path}/*')
    logging.info(f'cleaned {output_path}')
    # create rttm list and wav list
    rttms, wavs = [], [] 
    for wav in glob(input_path+"/*.wav"):
        file = wav.replace(".wav", "")
        rttms.append(file+".rttm")
        wavs.append(file+".wav")
      
    with open(f"{output_path}/rttm_list.txt", "w", encoding="utf-8") as tmp:
        tmp.write("\n".join(rttms))
        logging.info(f'saved: {output_path}/rttm_list.txt')
        
    with open(f"{output_path}/wav_list.txt", "w", encoding="utf-8") as tmp:
        tmp.write("\n".join(wavs))
        logging.info(f'saved: {output_path}/wav_list.txt')
        
def create_nemo_data_format(path):
    logging.info(f'create msdd_data.json')
    os.system(f'python nemo/scripts/speaker_tasks/pathfiles_to_diarize_manifest.py \
        --paths2audio_files="{path}/wav_list.txt" \
        --paths2rttm_files="{path}/rttm_list.txt" \
        --manifest_filepath="{path}/msdd_data.json"')
    
    logging.info(f'create msdd train dataset')
    os.system(f'python nemo/scripts/speaker_tasks/create_msdd_train_dataset.py \
        --input_manifest_path="{path}/msdd_data.json" \
        --output_manifest_path="{path}/msdd_data.50step.json" \
        --pairwise_rttm_output_folder="{path}" \
        --window 0.5 \
        --shift 0.25 \
        --step_count 50')

def prepare_data(input_path, output_path):
    create_rttmlist_and_wavlist(
        input_path=input_path,
        output_path=output_path
    )
    create_nemo_data_format(
        path=output_path
    )

if __name__ == "__main__":
    input_path = "datas/simulated_datas"
    output_path = "datas/nemo_datas"
    datasets = ["train", "valid", "test"]
    
    for set in tqdm(datasets):
        in_path = f'{input_path}/{set}'
        out_path = f'{output_path}/{set}'
        
        prepare_data(
            input_path=in_path,
            output_path=out_path
        )