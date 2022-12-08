def convert_rttm_to_segments(input_path, output_path):
    rttm_path = f'{input_path}/rttms'
    wav_path = f'{input_path}/wavs'
    os.system(f"rm -r {output_path}/*")
    for rttm_file in os.listdir(rttm_path):
        abs_rttm_path = os.path.join(rttm_path, rttm_file)
        abs_wav_path = os.path.join(wav_path, rttm_file.replace(".rttm", ".wav"))
        
        header = ["type", "wav_name", "o1", "offset", "duration", "o2", "o3", "speaker","o4", "o5"]
        rttm = pd.read_csv(abs_rttm_path, sep="\s+", names=header)
        wav, sr = librosa.load(abs_wav_path, sr = 16000)

        spk2utt = {}
        for index in rttm.index:
            if rttm["speaker"][index] not in spk2utt:
                spk2utt[rttm["speaker"][index]] = []
            else:
                start = int(rttm["offset"][index] * sr)
                end = int((rttm["offset"][index] + rttm["duration"][index]) * sr)
                spk2utt[rttm["speaker"][index]].append(wav[start:end])

        for spk, wavs in spk2utt.items():
            spk_wavs = []
            tmp_path = os.path.join(output_path, spk)
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
            for wav in wavs:
                spk_wavs += list(wav)
            abs_path = os.path.join(tmp_path, f"{rttm_file}_{spk}.wav")
            sf.write(abs_path, spk_wavs, sr)
            print("saved: ", abs_path)