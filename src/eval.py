import json
import os
import logging
import random
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from coref_bucket_batch_sampler import BucketBatchSampler
from data import get_dataset
from metrics import CorefEvaluator, MentionEvaluator
from datasets import load_metric
from utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters, extract_clusters_for_decode
from utils import extract_clusters_from_text, mention_to_cluster, extract_clusters_from_text_causal
from conll import evaluate_conll, debug_evaluate_conll
from metrics import b_cubed, muc, ceafe
from utils import flatten_list_of_lists
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.eval_output_dir = args.output_dir
        self.tokenizer = tokenizer
        # self.metric = load_metric("coval") #todo: uncomment for eval

    def evaluate(self, model, prefix="", type="Dev", tb_writer=None, global_step=None, official=False):
        eval_dataset = get_dataset(self.args, tokenizer=self.tokenizer, evaluate=True, file_type=type)

        if self.eval_output_dir and not os.path.exists(self.eval_output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.eval_output_dir)

        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = BucketBatchSampler(self.args, eval_dataset, max_total_seq_len=self.args.max_total_seq_len, batch_size_1=True)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Examples number: %d", len(eval_dataset))
        model.eval()

        # post_pruning_mention_evaluator = MentionEvaluator()
        # mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        losses = defaultdict(list)
        doc_to_prediction = {}
        doc_to_subtoken_map = {}
        for (doc_key, subtoken_maps), batch in eval_dataloader:
            batch = tuple(tensor.to(self.args.device) for tensor in batch)
            input_ids, attention_mask, labels, gold_clusters = batch

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                loss_dict = {}
                loss_dict.update({f"{type}_loss": loss})
            # gen_size = labels.size()[1] + 1
            if type == "Test":
                gen_size = max(eval_dataset.labels_lengths) + 1
                # beam search 1 for greedy search.
                model_gen_out = model.generate(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               num_beams=self.args.num_beams,
                                               max_length=1023)
                print("input_ids")
                print(input_ids.size())
                print("attention_mask")
                print(attention_mask.size())
                print("labels")
                print(labels.size())
                # model_gen_out = model.generate(input_ids=input_ids,
                #                                attention_mask=attention_mask,
                #                                num_beams=self.args.num_beams)
                if self.args.causal_tokenization:
                    model_gen_text = self.tokenizer.decode(model_gen_out[0, :], clean_up_tokenization_spaces=False)
                    label_text = self.tokenizer.decode(labels[0, :], clean_up_tokenization_spaces=False)
                else:
                    model_gen_text = self.tokenizer.decode(model_gen_out[0, :], clean_up_tokenization_spaces=False)
                    label_text = self.tokenizer.decode(labels[0, :], clean_up_tokenization_spaces=False)

                gold_clusters = gold_clusters.cpu().numpy().squeeze()
                gold_clusters = extract_clusters(gold_clusters)
                gold_clusters = sorted([sorted(cluster, key=lambda y: (y[0], y[1])) for cluster in gold_clusters])
                N = len(gold_clusters)
                gold_clusters_from_label = extract_clusters_from_text_causal(label_text, N)
                # gold_clusters_from_label = sorted([sorted(cluster, key=lambda y: (y[0], y[1])) for cluster in gold_clusters_from_label])
                mention_to_gold_clusters = mention_to_cluster(gold_clusters_from_label)
                predicted_clusters = extract_clusters_from_text_causal(model_gen_text, N)
                # predicted_clusters = sorted([sorted(cluster, key=lambda y: (y[0], y[1])) for cluster in predicted_clusters])
                if len(gold_clusters_from_label) != len(predicted_clusters):
                    print("Error in", doc_key, len(gold_clusters_from_label), len(predicted_clusters))
                mention_to_predicted_clusters = mention_to_cluster(predicted_clusters)
                split_doc_key = doc_key.split("_")
                orig_doc_key, sub_doc_key_idx = "_".join(split_doc_key[:-1]), int(split_doc_key[-1])
                # print(doc_key)
                # print(orig_doc_key)
                # print("N", N)
                print("label_text")
                print(label_text)
                print("Generate")
                print(model_gen_text)
                # print("input")
                # print(self.tokenizer.decode(input_ids[0, :], clean_up_tokenization_spaces=False))
                # print("gold_clusters")
                # print(gold_clusters)
                print("gold_clusters_from_label")
                print(gold_clusters_from_label)
                print("predicted_clusters")
                print(predicted_clusters)
                # print("mention_to_gold_clusters")
                # print(mention_to_gold_clusters)
                # print("mention_to_predicted_clusters")
                # print(mention_to_predicted_clusters, flush=True)
                print("*" * 20)
                coref_evaluator.update(predicted_clusters, gold_clusters_from_label, mention_to_predicted_clusters,
                                       mention_to_gold_clusters)
                # coref_evaluator.update(gold_clusters_from_label, gold_clusters_from_label, mention_to_gold_clusters,
                #                        mention_to_gold_clusters)
                if orig_doc_key not in doc_to_prediction:
                    doc_to_prediction[orig_doc_key] = []
                    doc_to_subtoken_map[orig_doc_key] = []

                doc_to_prediction[orig_doc_key].append((sub_doc_key_idx, predicted_clusters))
                doc_to_subtoken_map[orig_doc_key].append((sub_doc_key_idx, subtoken_maps))

            if self.args.verbose:
                print("doc_key", doc_key, flush=True)

            if self.args.n_gpu > 1:
                loss_dict = {key: val.mean() for key, val in loss_dict.items()}

            for key, val in loss_dict.items():
                losses[key].append(val.item())

        for dk, pred_clusters_per_sent in doc_to_prediction.items():
            # print(f"Document key {dk}")
            pred_clusters_per_sent = [y[1] for y in sorted(pred_clusters_per_sent, key=lambda x: x[0])]
            clusters_per_sent_mapping = [y[1] for y in sorted(doc_to_subtoken_map[dk], key=lambda x: x[0])]
            # print("doc", dk, flush=True)
            # print("pred_clusters_per_sent", pred_clusters_per_sent, flush=True)
            if not len(pred_clusters_per_sent):
                continue
            lengths = [len(_) for _ in pred_clusters_per_sent]
            if min(lengths) != max(lengths):
                print("Error2 in", dk, lengths)
            merged_pred_clusters = [[] for _ in range(max(lengths))]
            for i, sent_cluster in enumerate(pred_clusters_per_sent):
                cluster_mapping = clusters_per_sent_mapping[i]
                for j, cluster in enumerate(sent_cluster):
                    new_cluster = []
                    if len(cluster):
                        for start, end in cluster:
                            if 0 <= start < len(cluster_mapping) and 0 <= end < len(cluster_mapping):
                                new_cluster.append((cluster_mapping[start], cluster_mapping[end]))
                    merged_pred_clusters[j].extend(new_cluster)
            doc_to_subtoken_map[dk] = flatten_list_of_lists([y[1] for y in sorted(doc_to_subtoken_map[dk], key=lambda x: x[0])])
            doc_to_prediction[dk] = merged_pred_clusters
        results = [(key, sum(val) / len(val)) for key, val in losses.items()]
            # if type == "Test":
        prec, rec, f1 = coref_evaluator.get_prf()
        results += [
            ("precision", prec),
            ("recall", rec),
            ("f1", f1)
        ]

        logger.info("***** Eval results {} *****".format(prefix))
        for key, values in results:
            if isinstance(values, float):
                logger.info(f"  {key} = {values:.3f}")
            else:
                logger.info(f"  {key} = {values}")
            if tb_writer is not None and global_step is not None:
                tb_writer.add_scalar(key, values, global_step)

        if self.eval_output_dir:
            output_eval_file = os.path.join(self.eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                if prefix:
                    writer.write(f'\n{prefix}:\n')
                for key, values in results:
                    if isinstance(values, float):
                        writer.write(f"{key} = {values:.3f}\n")
                    else:
                        writer.write(f"{key} = {values}\n")

        results = OrderedDict(results)
        results["experiment_name"] = self.args.experiment_name
        results["data"] = prefix
        with open(os.path.join(self.args.output_dir, "results.jsonl"), "a+") as f:
            f.write(json.dumps(results) + '\n')

        if official:
            with open(os.path.join(self.args.output_dir, "preds.jsonl"), "w") as f:
                f.write(json.dumps(doc_to_prediction) + '\n')
                f.write(json.dumps(doc_to_subtoken_map) + '\n')

            if self.args.conll_path_for_eval is not None:
                conll_func = debug_evaluate_conll if self.args.store_pred else evaluate_conll
                conll_results = conll_func(self.args.conll_path_for_eval, doc_to_prediction, doc_to_subtoken_map)
                if len(conll_results):
                    official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
                    logger.info('Official avg F1: %.4f' % official_f1)

        return results
