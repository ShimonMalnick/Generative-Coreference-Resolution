import json
import logging
import os
import pickle
from collections import namedtuple

import numpy as np
import torch

from consts import SPEAKER_START, SPEAKER_END, NULL_ID_FOR_COREF
from utils import flatten_list_of_lists, flatten_list_of_lists_by_num
from torch.utils.data import Dataset
from utils import extract_clusters_from_text, extract_clusters_from_text_causal

CorefExample = namedtuple("CorefExample", ["token_ids", "clusters", "labels"])

logger = logging.getLogger(__name__)


class CorefDataset(Dataset):
    def __init__(self, args, file_path, tokenizer, user, max_seq_length=-1, causal_tokenization=False, file_type="train"):
        self.user = user
        self.file_type = file_type
        self.tokenizer = tokenizer
        logger.info(f"Reading dataset from {file_path}")
        # examples, self.max_mention_num, self.max_cluster_size, self.max_num_clusters = self._parse_jsonlines(file_path)
        examples, self.max_mention_num, self.max_cluster_size, self.max_num_clusters = self._parse_jsonlines_by_sentences(file_path, sent_num=args.sent_num, sentence_step=args.sent_step)
        self.max_seq_length = max_seq_length
        if causal_tokenization:
            self.examples, self.input_ids_lengths, self.labels_lengths, self.num_examples_filtered = self.__causal_tokenization(examples)
        else:
            self.examples, self.input_ids_lengths, self.labels_lengths, self.num_examples_filtered = self._tokenize_example_to_token(examples)
        logger.info(
            f"Finished preprocessing Coref dataset. {len(self.examples)} examples were extracted, {self.num_examples_filtered} were filtered due to sequence length.")

    def _parse_jsonlines(self, file_path):
        examples = []
        max_mention_num = -1
        max_cluster_size = -1
        max_num_clusters = -1
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                doc_key = d["doc_key"]
                input_words = flatten_list_of_lists(d["sentences"])
                clusters = d["clusters"]
                max_mention_num = max(max_mention_num, len(flatten_list_of_lists(clusters)))
                max_cluster_size = max(max_cluster_size, max(len(cluster) for cluster in clusters) if clusters else 0)
                max_num_clusters = max(max_num_clusters, len(clusters) if clusters else 0)
                speakers = flatten_list_of_lists(d["speakers"])
                examples.append((doc_key, input_words, clusters, speakers))
        return examples, max_mention_num, max_cluster_size, max_num_clusters

    def _parse_jsonlines_by_sentences(self, file_path, sent_num, sentence_step):
        examples = []
        max_mention_num = -1
        max_cluster_size = -1
        max_num_clusters = -1
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                doc_key = d["doc_key"]
                input_words = flatten_list_of_lists_by_num(d["sentences"], sent_num, sentence_step)
                offsets = np.cumsum([0] + [len(lst) for lst in input_words][:-1])
                all_speakers = flatten_list_of_lists_by_num(d["speakers"], sent_num, sentence_step)
                all_clusters = d["clusters"]

                for idx, sub_input_words in enumerate(input_words):
                    speakers = all_speakers[idx]
                    start = offset = int(offsets[idx])
                    end = len(sub_input_words) + start
                    sub_clusters = []
                    for cluster in all_clusters:
                        filtered_cluster = []
                        for mention_start, mention_end in cluster:
                            if start <= mention_start and mention_end < end:
                                filtered_cluster.append([mention_start - offset, mention_end - offset])
                        sub_clusters.append(filtered_cluster)
                    # print(doc_key, len(sub_clusters))
                    # print("-" * 100)
                    max_mention_num = max(max_mention_num, len(flatten_list_of_lists(sub_clusters)))
                    max_cluster_size = max(max_cluster_size, max(len(cluster) for cluster in sub_clusters) if sub_clusters else 0)
                    max_num_clusters = max(max_num_clusters, len(sub_clusters) if sub_clusters else 0)
                    examples.append((f"{doc_key}_{idx}", sub_input_words, sub_clusters, speakers, offset))
        return examples, max_mention_num, max_cluster_size, max_num_clusters


    def _tokenize_example_to_token(self, examples):
        coref_examples = []
        input_ids_lengths = []
        label_lengths = []
        num_examples_filtered = 0
        for doc_key, words, clusters, speakers, _ in examples:
            word_idx_to_start_token_idx = dict()
            word_idx_to_end_token_idx = dict()
            end_token_idx_to_word_idx = []  # for <s>

            token_ids = []
            last_speaker = None
            for idx, (word, speaker) in enumerate(zip(words, speakers)):
                if last_speaker != speaker:
                    speaker_prefix = self.tokenizer.encode(f"[{speaker}]", add_special_tokens=False)
                    last_speaker = speaker
                else:
                    speaker_prefix = []
                for _ in range(len(speaker_prefix)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(speaker_prefix)
                word_idx_to_start_token_idx[idx] = len(token_ids)
                print("word", word, idx)
                tokenized = self.tokenizer.encode(" " + word, add_special_tokens=False)
                print("tokenized", tokenized)
                print("len(token_id)", len(token_ids))
                for _ in range(len(tokenized)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(tokenized)
                print("len(token_id)", len(token_ids))
                word_idx_to_end_token_idx[idx] = len(token_ids) - 1

            if 0 < self.max_seq_length < len(token_ids):
                num_examples_filtered += 1
                continue

            print("clusters", clusters)
            new_clusters = [
                [(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for
                cluster in clusters]
            print("new_clusters", new_clusters)
            for cluster in clusters:
                for start, end in cluster:
                    print("mention",words[start: end+1], "start", start, "end", end)
            # new_clusters = [sorted(cluster, key=lambda x: x[0]) for cluster in new_clusters]
            # new_clusters.sort(key=lambda y: y[0][0])

            # save cluster per token index
            token_index_to_cluster_idx = dict()
            dup_count_in_cluster = dict()
            for cluster_idx, cluster in enumerate(new_clusters):
                for start, end in cluster:
                    for token_idx in range(start, end + 1):
                        if token_idx not in token_index_to_cluster_idx:
                            token_index_to_cluster_idx[token_idx] = []
                            dup_count_in_cluster[token_idx] = dict()
                        token_index_to_cluster_idx[token_idx].append(cluster_idx + 1)
                        if (cluster_idx + 1) in dup_count_in_cluster[token_idx]:
                            dup_count_in_cluster[token_idx][cluster_idx + 1] += 1
                        else:
                            dup_count_in_cluster[token_idx][cluster_idx + 1] = 0
            labels = []
            for token_idx in range(len(token_ids)):
                # create cluster label - per token
                if token_idx in token_index_to_cluster_idx:
                    token_rep = "("
                    for cluster_idx in token_index_to_cluster_idx[token_idx]:
                        if dup_count_in_cluster[token_idx][cluster_idx]:
                            token_rep += f"{cluster_idx}.{dup_count_in_cluster[token_idx][cluster_idx]},"
                            dup_count_in_cluster[token_idx][cluster_idx] -= 1
                        else:
                            token_rep += f"{cluster_idx},"
                    token_rep = token_rep[:-1] + ")"
                    labels.append(token_rep)
                else:
                    # Epsilon cluster idx is zero.
                    labels.append("0")

            label_str = " ".join(labels)
            new_labels = self.tokenizer.encode(label_str, add_special_tokens=False)
            # predicted_clusters = extract_clusters_from_text(label_str)
            print("label string", label_str)
            print("gold clusters", new_clusters)
            print("Input sentence", self.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False))
            print("-" * 200)
            # print("predicted clusters", predicted_clusters)
            # exit(0)
            input_ids_lengths.append(len(token_ids))
            label_lengths.append(len(new_labels))
            coref_examples.append(((doc_key, end_token_idx_to_word_idx), CorefExample(token_ids=token_ids,
                                                                                      clusters=clusters,
                                                                                      labels=new_labels)))
        exit(0)
        return coref_examples, input_ids_lengths, label_lengths, num_examples_filtered

    def __causal_tokenization(self, examples):
        coref_examples = []
        input_ids_lengths = []
        label_lengths = []
        num_examples_filtered = 0
        removed_docs = []
        shit_dict = dict()
        for doc_key, words, clusters, speakers, offset in examples:
            word_idx_to_start_token_idx = dict()
            word_idx_to_end_token_idx = dict()
            end_token_idx_to_word_idx = []  # for <s>
            word_idx_to_word_idx = []  # for <s>

            token_ids = []
            label = []
            last_speaker = None
            # relative start, end (not absolute)
            new_clusters = [[(start, end) for start, end in cluster] for cluster in clusters]

            for idx, (word, speaker) in enumerate(zip(words, speakers)):
                if last_speaker != speaker:
                    speaker_prefix = self.tokenizer.encode(f"[{speaker}]", add_special_tokens=False)
                    last_speaker = speaker
                else:
                    speaker_prefix = []
                # Add speaker tokens
                for _ in range(len(speaker_prefix)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(speaker_prefix)
                word_idx_to_start_token_idx[idx] = len(token_ids)
                tokenized = self.tokenizer.encode(" " + word, add_special_tokens=False)
                for _ in range(len(tokenized)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(tokenized)
                word_idx_to_end_token_idx[idx] = len(token_ids)  # old_seq_len + 1 (for <s>) + len(tokenized_word) - 1 (we start counting from zero) = len(token_ids)

                # save cluster per word index
                word_index_to_cluster_idx = dict()
                dup_count_in_cluster = dict()
                for cluster_idx, cluster in enumerate(new_clusters):
                    for start, end in cluster:
                        for word_idx in range(start, end + 1):
                            if word_idx not in word_index_to_cluster_idx:
                                word_index_to_cluster_idx[word_idx] = []
                                dup_count_in_cluster[word_idx] = dict()
                            word_index_to_cluster_idx[word_idx].append(cluster_idx)
                            if (cluster_idx) in dup_count_in_cluster[word_idx]:
                                dup_count_in_cluster[word_idx][cluster_idx] += 1
                            else:
                                dup_count_in_cluster[word_idx][cluster_idx] = 0

                if idx in word_index_to_cluster_idx:
                    word_str = ""
                    for cluster_idx in word_index_to_cluster_idx[idx]:
                        if dup_count_in_cluster[idx][cluster_idx]:
                            word_str += f"{cluster_idx}.{dup_count_in_cluster[idx][cluster_idx]},"
                            dup_count_in_cluster[idx][cluster_idx] -= 1
                        else:
                            word_str += f"{cluster_idx},"
                    word_str = "(" + word_str[:-1] + ")"
                else:
                    word_str = f" {word}"
                label.append(word_str)
                word_idx_to_word_idx.append(idx + int(offset))

            if 0 < self.max_seq_length < len(token_ids):
                num_examples_filtered += 1
                removed_docs.append(doc_key)
                continue

            label_str = " ".join(label)
            # print("label", label_str)
            # print("clusters", new_clusters)
            clusters_with_offset = []
            for cluster in new_clusters:
                c = []
                for start, end in cluster:
                    c.append((start + offset, end + offset))
                clusters_with_offset.append(c)
            # print("clusters with offset", clusters_with_offset)
            # print("offset", offset)
            # print("-"*100)
            labels = self.tokenizer.encode(label_str, add_special_tokens=False)
            for i in range(len(new_clusters)):
                if not len(new_clusters[i]):
                    new_clusters[i].append((-1, -1))

            split_doc_key = doc_key.split("_")
            orig_doc_key, sub_doc_key_idx = "_".join(split_doc_key[:-1]), int(split_doc_key[-1])
            if orig_doc_key not in shit_dict:
                shit_dict[orig_doc_key] = len(new_clusters)

            predicted_clusters = extract_clusters_from_text_causal(self.tokenizer.decode(labels, clean_up_tokenization_spaces=False), len(new_clusters))
            assert len(new_clusters) == len(predicted_clusters)
            input_ids_lengths.append(len(token_ids))
            label_lengths.append(len(labels))
            coref_examples.append(((doc_key, word_idx_to_word_idx),
                                   CorefExample(token_ids=token_ids,
                                                clusters=new_clusters,
                                                labels=labels)))
        with open(f"removed_docs_{self.file_type}_{self.user}.pkl", "wb") as f:
            pickle.dump(removed_docs, f)
        return coref_examples, input_ids_lengths, label_lengths, num_examples_filtered

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def pad_clusters_inside(self, clusters):
        return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (self.max_cluster_size - len(cluster)) for cluster
                in clusters]

    def pad_clusters_outside(self, clusters):
        return clusters + [[]] * (self.max_num_clusters - len(clusters))

    def pad_clusters(self, clusters):
        clusters = self.pad_clusters_outside(clusters)
        clusters = self.pad_clusters_inside(clusters)
        return clusters

    def pad_batch(self, batch, max_length, is_bart):
        padded_batch = []
        label_max_length = max(self.labels_lengths)
        input_ids_max_length = max(self.input_ids_lengths)
        for example in batch:
            encoded_dict = self.tokenizer.encode_plus(example[0],
                                                      add_special_tokens=False,
                                                      pad_to_max_length=True,
                                                      max_length=max_length,
                                                      return_attention_mask=True,
                                                      return_tensors='pt')
            if is_bart:
                print("Is Bart!", flush=True)
                label_dict = self.tokenizer.encode_plus(example[2],
                                                          add_special_tokens=True,
                                                          truncation=True,
                                                          max_length=1024,
                                                          return_attention_mask=False,
                                                          return_tensors='pt')
            else:
                print("Not Bart!", flush=True)
                label_dict = self.tokenizer.encode_plus(example[2],
                                                        add_special_tokens=True,
                                                        pad_to_max_length=True,
                                                        return_attention_mask=False,
                                                        return_tensors='pt')
            clusters = self.pad_clusters(example[1])
            example = (encoded_dict["input_ids"], encoded_dict["attention_mask"], label_dict["input_ids"], torch.tensor(clusters))
            padded_batch.append(example)
        tensored_batch = tuple(torch.stack([example[i].squeeze() for example in padded_batch], dim=0) for i in range(len(example)))
        return tensored_batch


def get_dataset(args, tokenizer, evaluate=False, file_type="train"):
    read_from_cache, file_path = False, ''
    if evaluate and os.path.exists(args.predict_file_cache):
       file_path = args.predict_file_cache
       read_from_cache = True
    elif (not evaluate) and os.path.exists(args.train_file_cache):
       file_path = args.train_file_cache
       read_from_cache = True

    if read_from_cache:
       logger.info(f"Reading dataset from {file_path}")
       with open(file_path, 'rb') as f:
           return pickle.load(f)

    file_path, cache_path = (args.predict_file, args.predict_file_cache) if evaluate else (args.train_file, args.train_file_cache)
    print(args, flush=True)
    coref_dataset = CorefDataset(args, file_path, tokenizer, max_seq_length=args.max_seq_length, causal_tokenization=args.causal_tokenization, file_type=file_type, user="eliav" if args.eliav else "malnick")
    with open(cache_path, 'wb') as f:
        pickle.dump(coref_dataset, f)

    return coref_dataset
