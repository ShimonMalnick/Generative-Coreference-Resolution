import json
import os
from datetime import datetime
from time import time
import git
import torch

from consts import NULL_ID_FOR_COREF


def flatten_list_of_lists(lst):
    return [elem for sublst in lst for elem in sublst]


def flatten_list_of_lists_by_num(lst, n, step):
    if n == -1:
        return flatten_list_of_lists(lst)
    new_lst = []
    curr = []
    idx = 0
    start = 0
    count = 0
    # print("total len", len(lst))
    while idx < len(lst):
        l = lst[idx]

        if not (count % n):
            if curr:
                new_lst.append(curr)
                # print("start", start)
                # print("count", count)
                # print("idx", idx)
                start += step
                count = 0
                idx = start
            curr = []

        for w in l:
            curr.append(w)
        idx += 1
        count += 1

    if curr:
        new_lst.append(curr)
    return new_lst


def mention_to_cluster(clusters):
    mentions_to_clusters = dict()
    for cluster in clusters:
        # print("mention_to_cluster", cluster)
        for mention in cluster:
            mentions_to_clusters[mention] = tuple(cluster)
    return mentions_to_clusters


def check_float(string):
    try:
        f = float(string)
        return True
    except Exception as e:
        return False


def extract_clusters_from_text_causal(text, max_clusters):
    split_text = text.split()
    clusters_memory = dict()
    for idx, word in enumerate(split_text):
        clusters = []
        # found cluster mark
        if word.startswith("(") and word.endswith(")"):
            clusters = [cluster_idx for cluster_idx in word.strip(")").strip("(").strip(" ").split(",") if check_float(cluster_idx)]

        for cluster_idx in clusters:
            if cluster_idx not in clusters_memory:
                clusters_memory[cluster_idx] = []
            clusters_memory[cluster_idx].append(idx)

    clusters_memory = [(cluster_idx, token_indices) for cluster_idx, token_indices in clusters_memory.items()]
    clusters_dict = dict()
    for cluster_idx, word_indices in clusters_memory:
        cluster = []
        prev = None
        start = None
        for index, word_index in enumerate(word_indices):
            if index == (len(word_indices) - 1):
                if prev == (word_index - 1):
                    cluster.append((start, word_index))
                else:
                    if start is not None and prev is not None:
                        cluster.append((start, prev))
                    cluster.append((word_index, word_index))
            else:
                if prev is None:
                    start = word_index
                    prev = word_index

                elif prev == (word_index - 1):
                    prev = word_index

                else:
                    cluster.append((start, prev))
                    start = word_index
                    prev = word_index
        try:
            cluster_idx = int(cluster_idx if "." not in cluster_idx else cluster_idx.split(".")[0])
            if cluster_idx not in clusters_dict:
                clusters_dict[cluster_idx] = cluster
            else:
                clusters_dict[cluster_idx] += cluster
        except Exception as e:
            continue

    clusters = []
    for i in range(max_clusters):
        if i in clusters_dict:
            clusters.append(clusters_dict[i])
        else:
            clusters.append([])
    return clusters


def extract_clusters_from_text(text):
    split_text = text.split()
    clusters_memory = dict()
    for idx, word in enumerate(split_text):
        # found cluster mark
        # if word.startswith("[") and word.endswith("]"):
            # clusters = [cluster_idx for cluster_idx in word.strip(" ").strip("]").split(",") if check_float(cluster_idx)]
        clusters = [cluster_idx for cluster_idx in word.split(",") if check_float(cluster_idx)]
        for cluster_idx in clusters:
            if cluster_idx == "0":
                continue
            elif cluster_idx not in clusters_memory:
                clusters_memory[cluster_idx] = []
            clusters_memory[cluster_idx].append(idx)

    clusters_memory = [(cluster_idx, token_indices) for cluster_idx, token_indices in clusters_memory.items()]
    clusters_dict = dict()
    for cluster_idx, token_indices in clusters_memory:
        cluster = []
        prev = None
        start = None
        for index, token_idx in enumerate(token_indices):
            if index == (len(token_indices) - 1):
                if prev == (token_idx - 1):
                    cluster.append((start, token_idx))
                else:
                    if start is not None and prev is not None:
                        cluster.append((start, prev))
                    cluster.append((token_idx, token_idx))
            else:
                if prev is None:
                    start = token_idx
                    prev = token_idx

                elif prev == (token_idx - 1):
                    prev = token_idx

                else:
                    cluster.append((start, prev))
                    start = token_idx
                    prev = token_idx
        try:
            cluster_idx = int(cluster_idx if "." not in cluster_idx else cluster_idx.split(".")[0])
            if cluster_idx not in clusters_dict:
                clusters_dict[cluster_idx] = cluster
            else:
                clusters_dict[cluster_idx] += cluster
        except Exception as e:
            continue

    clusters = [c[1] for c in sorted([(cluster_idx, cluster) for cluster_idx, cluster in clusters_dict.items()], key=lambda x: x[0])]
    return clusters


def extract_clusters(gold_clusters):
    gold_clusters = [tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters.tolist()]
    gold_clusters = [cluster for cluster in gold_clusters if len(cluster) >= 0]
    return gold_clusters


def extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[tuple(mention)] = gc
    return mention_to_gold


def extract_clusters_for_decode(mention_to_antecedent):
    mention_to_antecedent = sorted(mention_to_antecedent)
    mention_to_cluster = {}
    clusters = []
    for mention, antecedent in mention_to_antecedent:
        if antecedent in mention_to_cluster:
            cluster_idx = mention_to_cluster[antecedent]
            clusters[cluster_idx].append(mention)
            mention_to_cluster[mention] = cluster_idx
        else:
            cluster_idx = len(clusters)
            mention_to_cluster[mention] = cluster_idx
            mention_to_cluster[antecedent] = cluster_idx
            clusters.append([antecedent, mention])
    clusters = [tuple(cluster) for cluster in clusters]
    return clusters, mention_to_cluster


def mask_tensor(t, mask):
    t = t + ((1.0 - mask.float()) * -10000.0)
    t = torch.clamp(t, min=-10000.0, max=10000.0)
    return t


def write_meta_data(output_dir, args):
    output_path = os.path.join(output_dir, "meta.json")
    repo = git.Repo(search_parent_directories=True)
    hexsha = repo.head.commit.hexsha
    ts = time()
    print(f"Writing {output_path}")
    with open(output_path, mode='w') as f:
        json.dump(
            {
                'git_hexsha': hexsha,
                'args': {k: str(v) for k, v in args.__dict__.items()},
                'date': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            },
            f,
            indent=4,
            sort_keys=True)
        print(file=f)
