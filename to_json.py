"""
This script coverts the data in dev.csv, test.csv, and train.csv into a
sepearte json file for each dataset. Since each file contains three columns, 
the json file will contain a list of entries, where each entry is a dictionary
with three keys: "code_switched", "arabic", and "english". The value of each 
key is the text in the corresponding column.
"""

import json

import pandas as pd


def main():
    """Convert the data in dev.csv, test.csv, and train.csv into a json file."""
    ds_types = ["dev", "test", "train"]
    for ds_type in ds_types:
        file = f"data/{ds_type}.csv"
        df = pd.read_csv(file, encoding="utf-8")
        df = df.to_dict(orient="records")

        # write json and take into account the encoding
        with open(f"data/{ds_type}.json", "w", encoding="utf-8") as f:
            json.dump(df, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
