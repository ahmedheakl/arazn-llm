"""This script is to evaluate the models results"""
import pandas as pd
import sacrebleu
from tqdm import tqdm

translation_path = "translations/llama3-ara.csv"

df = pd.read_csv(translation_path)
tot = 0
bert_p = 0
bert_b = 0
meteor = 0
eed = 0
bert_f1 = 0
for record in tqdm(df.to_dict(orient='records')):
    reference = record['target']
    hypothesis = record['generated']
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference], tokenize="intl",
                                   smooth_method='add-k',
                                   lowercase=True)
    tot += bleu.score
    bert_f1 += float(record['bert_f1'])
    bert_b += float(record['bert_r'])
    bert_p += float(record['bert_p'])
    meteor += float(record['meteor_score'])
    eed += float(record['eed_score'])

print(len(df))
print(f"SacreBLEU score: {tot/len(df):.2f}")
print(f"BertScore F1: {bert_f1/len(df):.2f}")
print(f"BertScore precision: {bert_p/len(df):.2f}")
print(f"BertScore recall: {bert_b/len(df):.2f}")
print(f"Meteor score: {meteor/len(df):.2f}")
print(f"EED score: {eed/len(df):.2f}")