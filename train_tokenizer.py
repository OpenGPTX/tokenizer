
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import random
import os
import json
import zstandard as zstd
import argparse
from transformers import GPT2TokenizerFast
from dataloader import generator_all


def train_tokenizer(data_conf, eod_token, vocab_size, initial_vocab_file, initial_merge_file, save_dir):
    # initialize tokenizer with pretrained GPT2Tokenizer
    print("Initialize tokenizer")
    initial_tokenizer = GPT2TokenizerFast(
        initial_vocab_file,
        initial_merge_file,
        errors='replace',
        max_len=None
    )
    print("Train tokenizer")
    tokenizer = initial_tokenizer.train_new_from_iterator(generator_all(data_conf), vocab_size, new_special_tokens=[eod_token])
    print("Save tokenizer")
    tokenizer.save_pretrained(save_dir)
    print("Done")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT2 BPE tokenizer.")
    parser.add_argument("--data_conf", help="The path to the configuration json file for the dataset", required=True)
    parser.add_argument("--batch_size", help="Batch size for the fetched text", default=1)
    parser.add_argument("--save_dir", type=str, help="Save tokenizer to this directory")
    parser.add_argument("--initial_vocab_file", type=str, help="Initial vocab file")
    parser.add_argument("--initial_merge_file", type=str, help="Initial merge file")
    parser.add_argument("--vocab_size", type=int, help="Size of vocabulary")
    parser.add_argument("--eod_token", type=str, default="<|endoftext|>", help="End-Of-Document token")
    args = parser.parse_args()

    with open(args.data_conf,'r') as j:
        data_conf = json.load(j)
		
    train_tokenizer(
        data_conf=data_conf,
        eod_token=args.eod_token,
        vocab_size=args.vocab_size,
        initial_vocab_file=args.initial_vocab_file,
        initial_merge_file=args.initial_merge_file,
        save_dir=args.save_dir,
    )  
