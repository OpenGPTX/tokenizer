from datasets import load_from_disk
import json
import random
import argparse


def datasets_sampler(dataset_path, percentage, random_state=42):
    """
    This function returns sampled datasets from the pre-defined configurations.
    """
    #Read datasets (the try excpet is because: some of the data are not splitted into train/test (?), so this is a work around)
    try:
        dataset = load_from_disk(dataset_path)['train']
    except KeyError:
        dataset = load_from_disk(dataset_path)

    #sample the dataset
    num_samples = int(len(dataset)*percentage)
    dataset = dataset.shuffle(seed=random_state).select(range(num_samples))

    #return the sampled dataset
    return dataset



def generator_all(config,random_state=42):
    """
    This function behaves as a generator to stream text from the sampled data loaders for training.
    The text is produced in a randomaized manner.
    """
    #define a word counter
    word_count = 0
    
    #define threshold word count 3.2 billion token (or words)
    threshold_word_count = 3e9 + 2e8
    
    #load all the sampled data loaders
    all_dataloaders = [iter(datasets_sampler(dataset_path, percentage)) for dataset_path, percentage in config.items()]
    
    #log the number of used words per datasets in total_num_words_per_dataset
    total_num_words_per_dataset = {dataset_path: 0 for dataset_path in config.keys()}
    
    #start streaming text until the threshold_word_count is reached
    while word_count <= threshold_word_count:
        #in each iteration, select a random dataset to stream from 
        random_idx = random.choice(range(len(all_dataloaders)))
        
        #stream from the datasets, considering that texts could exist under different column name (work around)
        if 'text' in next(all_dataloaders[random_idx]).keys():
            text = next(all_dataloaders[random_idx])['text']
            
        elif 'article' in next(all_dataloaders[random_idx]).keys():
            text = next(all_dataloaders[random_idx])['article']
        
        elif 'content' in next(all_dataloaders[random_idx]).keys():
            text = next(all_dataloaders[random_idx])['content']  
              
        #stream text
        #TODO: How to handle dataloaders at stop iteration 
        yield text 
        
        #change the dataloader in the next iteration
        random_idx = random.choice(range(len(all_dataloaders)))
        
        #update word_count and total_num_words_per_dataset by the meassured word count from the streamed text in this iteration
        measured_word_count = len(text.split(' '))
        word_count += measured_word_count
        total_num_words_per_dataset[list(config.keys())[random_idx]] += measured_word_count
        #TODO: update logging the word count after 300 iteration 
        print(f"total used number of words: {word_count}")
        print(f"distribution of words per datasets: {total_num_words_per_dataset}")
        
    #when streaming is over, save the total_num_words_per_dataset as json    
    with open('total_num_words_per_dataset.json','w') as j: 
        json.dump(total_num_words_per_dataset,j)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turns downloads from download_common_crawl.py into a Hugging Face dataset, split by language (language is identified using a FastText model). The dataset has a timestamp column for the time it was crawled, along with a url column and, of course, a text column.")
    parser.add_argument("--input_conf", help="The path to the configuration json file", required=True)
    args = parser.parse_args()

    with open(args.input_conf,'r') as j:
        configs = json.load(j)
    i = 0 
    for d in generator_all(configs):
        print("Next item: \n\n\n\n")
        print(d)

        
        

    print("Done")