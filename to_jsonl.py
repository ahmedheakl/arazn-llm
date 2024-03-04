"""
This script coverts the data in dev.csv, test.csv, and train.csv into a
sepearte jsonl file for each dataset. Since each file contains three columns, 
the json file will contain a list of entries, where each entry is a dictionary
with three keys: "code_switched", "arabic", and "english". The value of each 
key is the text in the corresponding column.
"""

import json
import pandas as pd

DEV_CSV = "data/dev.csv"
TEST_CSV = "data/test.csv"
TRAIN_CSV = "data/train.csv"


def main():
    """Read the data from the csv files and convert it into a jsonl file."""
    dev = pd.read_csv(DEV_CSV, encoding="utf-8")
    test = pd.read_csv(TEST_CSV, encoding="utf-8")
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8")

    dev_data = []
    for i in range(len(dev)):
        entry = {
            "code_switched": dev.iloc[i, 0],
            "arabic": dev.iloc[i, 1],
            "english": dev.iloc[i, 2],
        }
        dev_data.append(entry)

    test_data = []
    for i in range(len(test)):
        entry = {
            "code_switched": test.iloc[i, 0],
            "arabic": test.iloc[i, 1],
            "english": test.iloc[i, 2],
        }
        test_data.append(entry)

    train_data = []
    for i in range(len(train)):
        entry = {
            "code_switched": train.iloc[i, 0],
            "arabic": train.iloc[i, 1],
            "english": train.iloc[i, 2],
        }
        train_data.append(entry)

    with open("data/dev.jsonl", "w", encoding="utf-8") as f:
        for entry in dev_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open("data/test.jsonl", "w", encoding="utf-8") as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
