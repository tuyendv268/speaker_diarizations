import os
import librosa
import logging
import numpy as np
import random
from pydub import AudioSegment
import yaml
from tqdm import tqdm
from yaml.loader import SafeLoader
from multiprocessing import Pool

def get_utt(path):
    utts = os.listdir(path)
    utts = [os.path.join(path, utt) for utt in utts]
    return utts

def add_silence(wav, duration) -> AudioSegment:
    """
        concat audio with silence
    """
    
    silent = AudioSegment.silent(duration=duration)
    wav = wav + silent
    
    return wav

def concat_two_audio(wav_1, wav_2, overlap_rate, prev_wav) -> AudioSegment:
    """
    param:
        wav_1: first wav
        wav_2 : second wav
        overlap_rate: overlap rate for concat two audio 
        prev_overlap: previous audio
    """
    overlap = round(overlap_rate*len(prev_wav) / 1000, 2)
    
    output = wav_1.overlay(wav_2, len(wav_1) - overlap*1000)
    offset = (len(wav_2) - overlap*1000)/1000
    if offset > 0:
        output = output + wav_2[overlap*1000:]
    
    return output, offset, overlap

def rttm_to_file(path, rttms):
    """save rttm content to file

    Args:
        path (str): _description_
        rttms (list): _description_
    """
    with open(path, "w", encoding="utf-8") as tmp:
        for rttm in rttms:
            line = f'{rttm["type"]}\t{rttm["wav"]}\t{rttm["chnl"]}\t{rttm["offset"]}\t{rttm["duration"]}\t{rttm["ortho"]}\t{rttm["stype"]}\t{rttm["speaker"]}\t{rttm["conf"]}\t{rttm["slat"]}\t'
            tmp.write(line+"\n")
    
def gen_mixture_wav(spk2utt, mixture_index, speakers, overlap_prob, concat_mode, silence_duration, overlap_rate, overlap_rate_prob):
    """generate simulated audio
    Args:
        spk2utt (dict): speaker to utterance
        mixture_index (list): index for simulation
        speakers(list): speakers
        concat_mode(int)
        overlap_rate(float): overlap rate
        overlap_rate_prob(list) : probability for each overlap rate
    Returns:
        wavs:
        rttms:
    """
    # mode = [0, 1]
    # overlap_prob = [0.6, 0.4]
    rttms = []
    speakers_idx = {spk:0 for spk in speakers}
    wavs = None
    prev_spk = None
    offset = 0
    for i, spk_idx in enumerate(mixture_index):
        rttm = {}
        speaker = speakers[spk_idx]
        wav_path = spk2utt[speaker][speakers_idx[speaker]]
        tmp_wav = AudioSegment.from_wav(wav_path)
        
        if wavs == None:
            wavs = tmp_wav
            offset = 0
            offset_tmp = len(tmp_wav)/1000
            # print(f"{i} mode: init - off_set: {offset} - offset_tmp: {offset_tmp} - duration {len(tmp_wav)/1000}")
        else:
            """
            random concat mode
            if choice == 1: overlap two audio
            if choice == 0: append silience
            """
            choice = random.choices(concat_mode, overlap_prob)[0]
            # if len(prev_wav) < 500 or len(tmp_wav) < 500:
            #     choice = 0
            if choice == 1 and prev_spk != spk_idx:
                overlap_rate_rnd = random.choices(overlap_rate, overlap_rate_prob)[0]
                wavs, ofs, overlap = concat_two_audio(wavs, tmp_wav, overlap_rate_rnd, prev_wav)
                if ofs > 0:
                    offset -= overlap
                offset_tmp = len(tmp_wav)/1000
                # print(f"{i} mode: overlap - off_set: {offset} -overlap: {overlap} - offset_tmp: {offset_tmp} - duration {len(tmp_wav)/1000}")
            else:
                silence = random.choice([i for i in range(300, 1500, 100)])
                wavs = add_silence(wavs, silence)
                wavs += tmp_wav
                offset += silence/1000
                offset_tmp = len(tmp_wav)/1000
                # print(f"{i} mode: concat - off_set: {offset} - offset_tmp: {offset_tmp} - slince : {silence/1000} - duration: {len(tmp_wav)/1000}")

                # continue
        rttm["duration"] = len(tmp_wav)/1000
        rttm["type"] = "SPEAKER"
        rttm["wav"] = "--".join(speakers)
        rttm["chnl"] = 1
        rttm["offset"] = round(offset,2)
        rttm["ortho"]="<NA>"
        rttm["stype"]="<NA>"
        rttm["speaker"] = speaker
        rttm["conf"]="<NA>"
        rttm["slat"]="<NA>"
        offset += offset_tmp
        prev_wav = tmp_wav
        prev_spk = spk_idx
        speakers_idx[speaker] += 1
        rttms.append(rttm)
    # print(rttms)
    return wavs, rttms

def save_data(wavs, rttms, out_path):
    """save wav and rttm

    Args:
        wavs (_type_): _description_
        rttms (_type_): _description_
        out_path (_type_): _description_
    """
    wav_name = rttms[0][0]["wav"]
    idx = 0
    for wav, rttm in zip(wavs, rttms):
        wav.export(f"{out_path}/{wav_name}_{idx}.wav", format="wav", codec='pcm_s16le')
        # logging.info(f'saved: {out_path}/{wav_name}_{idx}.wav')
        
        for i in rttm:
            i["wav"] = f"{wav_name}_{idx}"
        rttm_to_file(f"{out_path}/{wav_name}_{idx}.rttm", rttm)
        # logging.info(f'saved: {out_path}/{wav_name}_{idx}.rttm')
        idx += 1
    # print(f'saved: {out_path}/{wav_name}_{idx}.rttm')

def split_wav(wavs, rttms):
    """split audio with num utterance
        step = 100: each rttm and wav file containing 100 utterance
    Args:
        wavs (_type_): _description_
        rttms (_type_): _description_

    Returns:
        wav_out: list
        rttm_out: list
    """
    wav_out = []
    step = 100 
    rttm_out = [rttms[i:i+step] for i in range(0, len(rttms), step)]
    
    for rttm in rttm_out:
        start = rttm[0]["offset"]
        end = rttm[-1]["offset"] + rttm[-1]["duration"]
        wav_out.append(wavs[start*1000:end*1000])
        for i in range(len(rttm)):
            rttm[i]["offset"] = round(rttm[i]["offset"] - start,2)
    return wav_out, rttm_out

def gen_audio(inp_path, out_path, config):
    num_speaker = random.choices(config["num_speakers"], config["prob"])[0]
    # random speaker
    # print(f"num_speaker: {num_speaker}")
    speaker_list = os.listdir(inp_path)
    # print(speaker_list)
    """
        sampling speakers
    """
    speakers = random.sample(speaker_list, num_speaker)
    # print(f"speakers: {speakers}")
    
    spk2utt = dict.fromkeys(speakers, 0)
    mixture_index = []
    for index, speaker in enumerate(speakers):
        abs_path = os.path.join(inp_path, speaker)
        spk2utt[speaker]=get_utt(abs_path)
        mixture_index += [index] * len(spk2utt[speaker])
    random.shuffle(mixture_index)
    # print(mixture_index)
    
    wav_name = "--".join(speakers)
    
    res, rttms = gen_mixture_wav(
        spk2utt, 
        mixture_index, 
        speakers, 
        overlap_prob=config["overlap_prob"], 
        concat_mode=config["concat_mode"],
        silence_duration=config["silence_duration"],
        overlap_rate=config["overlap_rate"],
        overlap_rate_prob=config["overlap_rate_prob"]
    )
    
    # print(rttms)
    # res.export(f"{out_path}/{wav_name}.wav", format="wav") 
    # rttm_to_file(f"{out_path}/{wav_name}.rttm", rttms)
    wavs, rttms = split_wav(res, rttms)
    save_data(wavs, rttms, out_path)
    
def simulate_wav(inp_path, out_path, config, num_sample_per_process):
    # print(inp_path)
    for i in tqdm(range(0, num_sample_per_process)):
        try:
            gen_audio(inp_path, out_path, config)
        except:
            continue

def simulate_data_in_parallel(config, input_path, output_path, num_process, num_sample_per_process):
    argv = [[input_path, output_path, config, num_sample_per_process] for i in range(num_process)]
        
    pool = Pool(num_process)
    pool.starmap(simulate_wav, argv)
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    with open("simulation_config.yaml","r") as f:
        config = yaml.load(f, Loader=SafeLoader)
        
    num_process = 5
    num_sample_per_process = 1
    inp_path="datas/preprocessed_datas/valid"
    out_path="datas/simulated_datas/valid"

        
    simulate_data_in_parallel(
        config=config,
        input_path=inp_path,
        output_path=out_path,
        num_process=num_process,
        num_sample_per_process=num_sample_per_process
    )