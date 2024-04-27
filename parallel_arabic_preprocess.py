import os 
import re
from tqdm import tqdm

import pandas as pd

DATA_ROOT = "arabic-parallel"
NUM_ROWS = 24_000

def clean_text(text):
    # Remove special characters such as [U+202B]
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text.strip()

def main():
    data = {"arabic": "", "english": ""}
    df = pd.DataFrame(data, index=[0])

    files = [[f"{DATA_ROOT}/{folder}/{file}", folder] for folder in os.listdir(DATA_ROOT) for file in os.listdir(f"{DATA_ROOT}/{folder}")]

    for file, folder in tqdm(files):
        if folder != "Songs":
            file_df = pd.read_excel(file, header=None)
            file_df = file_df.iloc[:, :2]
            file_df.columns = ["arabic", "english"]
        else:
            file_df = pd.read_excel(file)
            file_df = file_df[["Egyptian Arabic Lyrics", "English Translation"]]
            file_df.columns = ["arabic", "english"]

        file_df = file_df.dropna()
        file_df['arabic'] = file_df['arabic'].apply(clean_text)
        file_df['english'] = file_df['english'].apply(clean_text)
        df = pd.concat([df, file_df])
            
    df = df.drop(index=0)
    df = df.reset_index(drop=True)

    df = df.iloc[:NUM_ROWS]

    output_path = f"arabic_parallel_{NUM_ROWS}.jsonl"
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Data with {NUM_ROWS} records written to {output_path}")


if __name__ == "__main__":
    main()

