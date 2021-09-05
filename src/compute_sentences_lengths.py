import seaborn as sns
from utils import flatten_list_of_lists_by_num
import matplotlib.pyplot as plt
import pandas as pd
from transformers import T5Tokenizer, BartTokenizer
import os
import json


def create_boxplot_stats(files_paths, tokenizer, num_sentences):
    tokenized_lengths = [[] for _ in num_sentences]
    for file_path in files_paths:
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                sentences = d["sentences"]
                for idx, num in enumerate(num_sentences):
                    new_sentences = flatten_list_of_lists_by_num(sentences, num)
                    for sent in new_sentences:
                        tokenized_len = 0
                        for word in sent:
                            tokenized_len += len(tokenizer.encode(word))
                        tokenized_lengths[idx].append(tokenized_len)
    dataframes = [pd.df(tokenized_lengths[i])] #to do: dataframes of the
    # distribution and box plots



def parse_jsonlines_max_lengths(file_path, tokenizer, num_sentences):
    with open(file_path, 'r') as f:
        for line in f:
            d = json.loads(line.strip())
            sentences = d["sentences"]

            max_lengths = []
            for num in num_sentences:
                new_sentences = flatten_list_of_lists_by_num(sentences, num)
                max_len = 0
                for sent in new_sentences:
                    tokenized_len = 0
                    for word in sent:
                        tokenized_len += len(tokenizer.encode(word, add_special_tokens=False))
                    max_len = max(tokenized_len, max_len)
                max_lengths.append(max_len)

    print(f"Finished file{file_path}")
    return max_lengths


def run_script(data_dir_path="/home/yandex/AMNLP2021/moreliav/coref"):
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base", cache_dir="cache")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", cache_dir="cache")
    tokenizers = [(t5_tokenizer, "T5"), (bart_tokenizer, "BART")]
    data_files = ["dev.jsonlines", "train.jsonlines", "test.jsonlines"]
    lines =[]
    num_sentences = [3, 5, 8, 10, 15, 20, 25, 30]
    for tokenizer, name in tokenizers:
        lines.append(name)
        max_global_values = [0 for _ in num_sentences]
        for file in data_files:
            lines.append("\t" + file.replace(".jsonlines", '') + ":")
            file_path = os.path.join(data_dir_path, file)
            max_values = parse_jsonlines_max_lengths(file_path, tokenizer,
                                                     num_sentences)

            for idx, num in enumerate(num_sentences):
                lines.append(f"Number Of Sentences:{num}")
                lines.append(f"Max Tokens: {max_values[idx]}")
        lines.append("\n")
        best_val = max([num_sentences[idx] for idx, num in enumerate(max_values) if num < 512])
        lines.append(f"For {name} model The maximal number of sentences with less than 512 tokens is {best_val}")
    with open("Max_sentences_lengths.txt", "w") as out_file:
        new_lined = [line + "\n" for line in lines]
        out_file.writelines(new_lined)


if __name__ == '__main__':
    run_script()