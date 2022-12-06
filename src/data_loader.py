from random import Random, randint
import torch
from torch.utils.data import DataLoader, random_split

class Dataset:

    def __init__(self, args, label_to_id = None, mode = 'train', lang = None):

        self.filename = args['train_data']
        self.device = args['device']
        self.batch_size = args['batch_size']

        #the dictionary of langauge and tuple
        self.data = []

        #dummy dataset
        if(lang is None):
            return 

        #read from file
        with open(self.filename, "r") as f:

            header = f.readline()
            #read the file
            for line in f:
                
                #split the line on tab
                line_list = line.rstrip().split('\t')
                
                #the hypothesis and premise
                x_premise = line_list[1]
                x_hypothesis = line_list[2]

                #check if language in dict
                if(line_list[-1] != lang):
                    continue

                #append the data
                if(mode == 'train'):
                    #TOKENS, LANGUAGE, GOLD_LABEL
                    self.data.append((x_hypothesis,x_premise,lang, label_to_id[line_list[0]]))
                else:
                    #TOKENS, LANGUAGE
                    self.data.append((x_hypothesis, x_premise, lang))

        #shuffle the data
        Random(args['seed']).shuffle(self.data)

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def __getitem__(self,index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

'''combines the data according to batch size'''
def combine(datasets, batch_size):
    '''
    Args:
        datasets: dictionary of datasets
        batch_size: batch size
    Return:
        final_dataset:
            batched data according to the language 
    '''
    data = []
    index = []
    for lang in datasets:
        index.append(0)

    #while not alloted all data
    while (len(datasets) > 0):

        #just take the data of batch size and repeat
        r = randint(0, len(index)-1)

        #the start index and end index
        start_index = index[r]
        end_index = index[r] + batch_size
        index[r] += batch_size

        #data len
        data_batch = datasets[r][start_index: end_index]

        #check for incomplete batch
        if(len(data_batch) < batch_size):
            for i in range(0, len(data_batch) - batch_size):
                data_batch.append(data_batch[-1])

        #delete the lang data if used all
        if(index[r] >= len(datasets[r])):
            del datasets[r]
            del index[r]
        
        #add the batch
        data.extend(data_batch)
    
    #return
    return data

'''
we would return the 2 dictionary of datasets 
'''
def get_dataset(args, label_to_id):
    '''
    Args:
        args: the configuration dictionary
        label_to_id: label to id dictionary
        mode: mode of the dataset
    Returns:
        train_dataset: train dataset instance
        val_dataset: splitted dataser
    '''
    #set the seed of torch
    torch.manual_seed(args['seed'])

    #get the dataset
    train_datasets = dict()
    for lang in args['train_lang']:
        train_datasets[lang] = Dataset(args, label_to_id, mode = 'train', lang = lang).data

    #create the val datasets
    val_datasets = dict()
    for lang in args['val_lang']:
        #get the train data
        data = train_datasets[lang]

        #train len
        train_size = int(len(data)*args['train_split'])
    
        #add to dictionaries
        train_datasets[lang] = data[0:train_size]
        val_datasets[lang] = Dataset(args, label_to_id, mode = 'train', lang = lang)
        val_datasets[lang].set_data(data[train_size+1:])

    #combine the datasets in the
    final_data = combine(list(train_datasets.values()), args['batch_size'])

    #final dataset
    train_dataset = Dataset(args, label_to_id, mode = 'train', lang = None)
    train_dataset.set_data(final_data)

    #return the corresponding data
    return train_dataset, val_datasets               