import argparse
import os
import subprocess


def parse_args():
    description_str = 'Run a Generative Coreference Resolution model'
    parser = argparse.ArgumentParser(description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_type', type=str, default="t5-base", help='model to run. either T5 or BART')
    parser.add_argument('--split_for_eval', type=str, default="dev", help='dev or test')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--gradient_acc', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--run_idx', type=int, default=0, help='index')
    parser.add_argument('--epochs', type=int, default=70, help='number of epochs')
    parser.add_argument('--max_length', type=int, default=512, help='max length')
    parser.add_argument('--num_beams', type=int, default=3, help='num beams')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='warmup steps')
    parser.add_argument('--do_train', dest='do_train', action='store_true', help="do_train flag")
    parser.add_argument('--model_name_or_path', type=str, default="t5-base", help="model checkpoint path or name")
    parser.add_argument('--output_dir_suffix', type=str, default="", help="output_dir_suffix")
    parser.add_argument('--no_cuda', dest='no_cuda', action='store_true', help="run on cpu if true")
    parser.add_argument("--store_pred", action="store_true", help="whether to store conll prediction file")
    parser.add_argument('--baseline', dest='baseline', action='store_true', help="baseline")
    parser.add_argument('--sent_num', type=int, help='num of sentence per paragraph beams')
    parser.add_argument('--sent_step', type=int, help='num of sentence between paragraph step')

    parser.set_defaults(do_train=False)
    parser.set_defaults(baseline=False)

    return parser.parse_args()


def run_script(args):
    baseline = "baseline_" if args.baseline else ""
    file_path = os.path.join("./slurm", f"{args.model_type}"
                                        f"{'_baseline_' if baseline else '_'}epochs_{args.epochs}_dropout_{args.dropout}_accumulation_{args.gradient_acc}_lr_{str(args.lr)}{args.data_suffix}_{args.output_dir_suffix}_sent_num_{args.sent_num}_sent_step_{args.sent_step}".replace(os.sep, "_"))
    model_suffix = args.model_type if "t5" in args.model_type else "bart-base"
    lines = ["#! /bin/sh"]
    lines.append(f"export OUTPUT_DIR=$(readlink -f ./output)")
    lines.append(f"export CACHE_DIR=$(readlink -f ./cache)")
    lines.append(f"export MODEL_DIR=$(readlink -f ./model)")
    lines.append(f"export DATA_DIR=$(readlink -f ../coref{args.data_suffix})")
    lines.append(f"export SPLIT_FOR_EVAL={args.split_for_eval}")

    lines.append("python run_coref.py \\")
    lines.append(f"--output_dir=$OUTPUT_DIR/" + f"{args.model_type}"
                                                f"{'_baseline_' if baseline else '_'}"
                                                f"epochs_{args.epochs}_dropout_{args.dropout}_accumulation_{args.gradient_acc}_"
                                                f"lr_{str(args.lr)}_{args.output_dir_suffix}_sent_num_"
                                                f"{args.sent_num}_sent_step_{args.sent_step} \\".replace(os.sep, "_"))
    lines.append(f"--cache_dir=$CACHE_DIR \\")
    lines.append(f"--model_type={args.model_type} \\")
    lines.append(f"--model_name_or_path={args.model_name_or_path} \\")
    lines.append(f"--tokenizer_name={args.model_type} \\")
    lines.append(f"--config_name={args.model_type} \\")
    lines.append(f"--train_file=$DATA_DIR/train.english.jsonlines \\")
    lines.append(f"--predict_file=$DATA_DIR/{args.split_for_eval}.english.jsonlines \\")
    if args.do_train:
        lines.append(f"--do_train \\")
    lines.append(f"--do_eval \\")
    lines.append(f"--num_train_epochs={args.epochs} \\")
    lines.append(f"--logging_steps=500 \\")
    lines.append(f"--save_steps=3000 \\")
    lines.append(f"--eval_steps=2000 \\")
    lines.append(f"--max_seq_length={args.max_length} \\")
    lines.append(f"--train_file_cache=$DATA_DIR/train.{model_suffix}_{args.max_length}_num_sent_{args.sent_num}_step_sent_{args.sent_step}.english.pkl \\")
    lines.append(f"--predict_file_cache=$DATA_DIR/{args.split_for_eval}.{model_suffix}_{args.max_length}_num_sent_{args.sent_num}_step_sent_{args.sent_step}.english.pkl \\")
    lines.append(f"--gradient_accumulation_steps=1 \\")
    lines.append(f"--normalise_loss \\")
    lines.append(f"--max_total_seq_len={args.max_length} \\")
    lines.append(f'--experiment_name="{model_suffix}_{baseline}'
                 f'_{args.dropout}_{args.gradient_acc}_{str(args.lr)}" \\')
    lines.append(f"--warmup_steps={args.warmup_steps} \\")
    lines.append(f"--adam_epsilon=1e-6 \\")
    lines.append(f"--head_learning_rate=3e-4 \\")
    lines.append(f"--learning_rate={args.lr} \\")
    lines.append(f"--adam_beta2=0.98 \\")
    lines.append(f"--weight_decay=0.01 \\")
    lines.append(f"--dropout_prob={args.dropout} \\")
    lines.append(f"--save_if_best \\")
    lines.append(f"--top_lambda=0.4 \\")
    lines.append(f"--conll_path_for_eval=$DATA_DIR/{args.split_for_eval} \\")
    lines.append(f"--tensorboard_dir=$OUTPUT_DIR/" + f"tb_{args.model_type}"
                                                     f"{'_baseline_' if baseline else '_'}epochs_{args.epochs}_dropout_{args.dropout}_accumulation_{args.gradient_acc}_lr_{str(args.lr)}_{args.output_dir_suffix}_sent_num_{args.sent_num}_sent_step_{args.sent_step} \\".replace(os.sep, "_"))
    lines.append(f"--num_beams {args.num_beams} \\")
    lines.append(f"--sent_num {args.sent_num} \\")
    lines.append(f"--sent_step {args.sent_step} \\")

    if args.no_cuda:
        lines.append(f"--no_cuda \\")
    if args.baseline:
        lines.append(f"--baseline \\")
    lines.append(f"--causal_tokenization \\")
    if args.store_pred:
        lines.append(f"--store_pred \\")

    with open(file_path + ".sh", "w") as script_file:
        script_file.write('\n'.join(lines) + '\n')
    return file_path


if __name__ == '__main__':
    args = parse_args()
    file_path = run_script(args)
    subprocess.run(["sh", file_path])
