"""This script converts an ASR dataset into a Hugging Face dataset.

The audio files in ArzEn_SpeechCorpus/recording contains a lot of text. These audio files
should be loaded and divide according to timestamps in ArzEn_SpeechCorpus/ASR_files/text.
Each line in "text" contains the following information:
<utterance-id> <transcription>, the utterance ID consists of [speaker_ID]-[corpus_ID]-[Recording_ID]_[timestamp_start]-[timestamp_end].
where the timestamp represented in 6 digits representing the time in seconds divided by 100. For example, 123456 represents 1234.56 seconds.

The path to a recording is ArzEn_SpeechCorpus/recordings/[corpus_ID]-[Recording_ID].wav
"""
import os
from tqdm import tqdm
from datasets import Dataset, Audio

ROOT_DIR = "ArzEn_SpeechCorpus"
OUTPUT_AUDIO_DIR = f"{ROOT_DIR}/audio"

def main():
    # Load the text file
    with open(f"{ROOT_DIR}/ASR_files/text") as f:
        lines = f.readlines()

    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    audio_files = os.listdir(f"{ROOT_DIR}/recordings")
    print(f"[INFO] Loaded {len(audio_files)} audio files")

    df = {"audio": [], "sentence": []}

    for i, line in enumerate(tqdm(lines)):
        utterance_id, transcription = line.split(" ", 1)

        _, corpus_id, recording_id, timestamp_end = utterance_id.split("-")
        recording_id, timestamp_start = recording_id.split("_")
        timestamp_start = int(timestamp_start) / 100
        timestamp_end = int(timestamp_end) / 100
        
        audio_file = f"{corpus_id}-{recording_id}.WAV"
        if audio_file not in audio_files:
            print(f"[ERROR] Audio file {audio_file} not found")
            continue
        
        new_audio_file = f"{OUTPUT_AUDIO_DIR}/{utterance_id}.wav"
        os.system(f"sox {ROOT_DIR}/recordings/{audio_file} {new_audio_file} trim {timestamp_start} ={timestamp_end}")

        df["audio"].append(new_audio_file)
        df['sentence'].append(transcription.strip())

    
    dataset = Dataset.from_dict(df)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))



if __name__ == "__main__":
    main()