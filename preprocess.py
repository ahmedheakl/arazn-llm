"""
Load code-switched, arabic, and english triplets. Each language is in a separate folder 
transcriptions, translations_EgyptianArabic, translations_English, and each is divided into 
three files dev.src, test.src, and train.src. This script loads the triplets into a pandas
dataframe and saves it as csv file.

Each .src file contains two columns, i.e. the ID and the text. The ID is not used in this
script. The text is loaded into a pandas dataframe and concatenated into a single dataframe
for each language. The three dataframes are then concatenated into a single dataframe.
"""

import re

import pandas as pd

TRANSCRIPTIONS = "data/transcriptions/"
TRANSLATIONS_EGYPTIAN_ARABIC = "data/translations_EgyptianArabic/"
TRANSLATIONS_ENGLISH = "data/translations_English/"
CODE_SWITCHED = "code_switched"
ENGLISH = "english"
ARABIC = "arabic"


def preprocess_text(text: str) -> str:
    """
    Text might contain [HES], [LAUGHTER] .. etc.
    This function removes these tags from the text.
    """

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def load_triplets():
    """Load the code-switched, arabic, and english triplets into a pandas dataframe.
    Only the text is loaded into the dataframe. The ID is not used.
    """
    folders = [TRANSCRIPTIONS, TRANSLATIONS_EGYPTIAN_ARABIC, TRANSLATIONS_ENGLISH]
    ds_types = ["dev", "test", "train"]
    extensions = ["src", "tgtEg", "tgtEn"]
    columns = [CODE_SWITCHED, ARABIC, ENGLISH]

    dev = pd.DataFrame()
    test = pd.DataFrame()
    train = pd.DataFrame()

    datasets = {
        "dev": [],
        "test": [],
        "train": [],
    }
    for ds_type in ds_types:
        for i, folder in enumerate(folders):

            extension = extensions[i]
            column = columns[i]
            file = f"{folder}{ds_type}.{extension}"
            df = pd.read_csv(file, sep="\t", header=None, names=["ID", column])
            df = df.drop(columns=["ID"])

            df[column] = df[column].apply(preprocess_text)

            datasets[ds_type].append(df)

    dev = pd.concat(datasets["dev"], axis=1)
    test = pd.concat(datasets["test"], axis=1)
    train = pd.concat(datasets["train"], axis=1)

    return dev, test, train


def save_triplets(dev: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame):
    """Save triplets as csv files"""
    dev.to_csv("data/dev.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    train.to_csv("data/train.csv", index=False)


def main():
    """Load triplets and save as csv files"""
    dev, test, train = load_triplets()
    save_triplets(dev, test, train)


if __name__ == "__main__":
    main()
