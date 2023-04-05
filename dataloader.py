
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import json
import zstandard as zstd
import argparse


def datasets_sampler(dataset_path, percentage, random_state=42):
    """
    This function streams sampled texts from datasets from the pre-defined configurations.

    """
    #Read datasets (the try excpet is because: some of the data are not splitted into train/test (?), so this is a work around)
    try:
        dataset = load_from_disk(dataset_path)['train']
    except KeyError:
        dataset = load_from_disk(dataset_path)

    #sample the dataset
    num_samples = int(len(dataset)*percentage)
    dataset = dataset.shuffle(seed=random_state).select(range(num_samples))

    #estimate the number of batches
    num_batches = len(dataset) // batch_size

    #iterate through texts of sampled datasets
    for d in dataset:
        try:
            yield d['text']
        except KeyError:
            yield d['article']


def generator_all(config):
    """
    This function behaves as a generator to stream text from data loaders for training.

    The text is produced in a sequential manner.
    """
    #load all pile data loaders
    all_dataloaders = [datasets_sampler(dataset_path, percentage) for dataset_path, percentage in config.items()]

    #stream text
    for each_dataloader in all_dataloaders:
        for each_text in each_dataloader:
            yield each_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turns downloads from download_common_crawl.py into a Hugging Face dataset, split by language (language is identified using a FastText model). The dataset has a timestamp column for the time it was crawled, along with a url column and, of course, a text column.")
    parser.add_argument("--input_conf", help="The path to the configuration json file", required=True)
    args = parser.parse_args()

    with open(args.input_conf,'r') as j:
        configs = json.load(j)

    for d in generator_all(configs):
        print("Next item: \n\n\n\n")
        print(d)

    print("Done")
