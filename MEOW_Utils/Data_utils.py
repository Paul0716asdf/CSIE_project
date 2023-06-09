import copy
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertModel, BertEmbeddings, BertPooler
from pandas import DataFrame
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from MEOW_Models.Kernel_model import BertWithoutEmbedding, ModelingQA, ModelingCLF
import torch

# other helper function
def get_balanced_df(df, column_name):
    value_count = df.value_counts(column_name)
    key_arr = value_count.keys()
    balanced_df = pd.DataFrame()

    min_num = 99999999

    for key in key_arr:
        min_num = min(min_num, value_count[key])

    for key in key_arr:
        tmp_df = df[df[column_name] == key]
        tmp_df = tmp_df.sample(n=min_num, random_state=42)
        balanced_df = pd.concat([balanced_df, tmp_df])

    balanced_df = shuffle(balanced_df)
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df

def squad_json_to_dataframe_train(input_file_path, record_path = ['data','paragraphs','qas','answers'],
                           verbose = 1):
    if verbose:
        print("Reading the json file")    
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file , record_path )
    m = pd.io.json.json_normalize(file, record_path[:-1])
    r = pd.io.json.json_normalize(file,record_path[:-2])
    
    #combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx  = np.repeat(m['id'].values,m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    main = pd.concat([ m[['id','question','context']].set_index('id'),js.set_index('q_idx')],1,sort=False).reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main

def count_the_TKbeg_and_TKend(context, ans_start, answer, tokenizer):
    # the parameters is context, answer_start, text in the dataframe
    # split the context from answer start, and get the token start
    if(ans_start == -1):
        return 0, 0
    
    s1 = context[0:ans_start]

    TKstart = len(tokenizer.tokenize(s1))
    TKanslen = len(tokenizer.tokenize(answer))
    TKend = TKstart + TKanslen - 1

    return TKstart+1, TKend+1 # +1 because of [CLS]

class layer_helper():
    def __init__(self, pretrained_module_name, device):
        self.device = device
        # all classification class is use this modeling_layer_clf for modeling
        self.kernel_model = BertWithoutEmbedding.from_pretrained(pretrained_module_name)
        self.kernel_model.config.output_hidden_states = True
        self.modeling_layer_qa = ModelingQA(hid_size = self.kernel_model.config.hidden_size, device = self.device)
    
    def get_kernel_model(self):
        return self.kernel_model

    def get_pooler_layer(self, individual):
        if individual:
            return copy.deepcopy(self.kernel_model.pooler)
        return self.kernel_model.pooler 

    def get_embedding_layer(self, individual = False):
        if individual:
            return copy.deepcopy(self.kernel_model.embeddings)
        return self.kernel_model.embeddings
    
    def get_modelings_layer_clf(self, individual_pooler = False):
        return ModelingCLF(pooler = self.get_pooler_layer(individual = individual_pooler), hidden_size = self.kernel_model.config.hidden_size, device = self.device)
    
    def get_modelings_layer_qa(self):
        return self.modeling_layer_qa
    
#### balanced dataset
#### this function can create the data we want to train
####------------------------------------------------------------------------
def create_CoLA_df(file_path, tokenizer = None, data_size = 0):
    train_df = pd.read_csv(file_path)
    if(data_size != 0):
        train_df = train_df[0:data_size]

    balanced_df = get_balanced_df(train_df, column_name='label')
    balanced_df = balanced_df.reset_index(drop=True)
    train_df = balanced_df
    
    if 'Unnamed: 0' in train_df.keys():
        train_df = train_df.drop('Unnamed: 0', axis=1)

    train_df['SEP_ind'] = train_df['context1'].apply(lambda x : len(tokenizer.tokenize(x))+1) # +1 is [CLS]
    train_df['context2'] = train_df['context2'].apply(lambda x : x if type(x) is str else '')

    train_df.to_csv('Dataset_infile\_CoLA.csv')

def create_MNLI_df(file_path, tokenizer = None, data_size = 0):
    train_df = pd.read_csv(file_path)
    if(data_size != 0):
        train_df = train_df[0:data_size]

    balanced_df = get_balanced_df(train_df, column_name='label')
    balanced_df = balanced_df.reset_index(drop=True)
    train_df = balanced_df
    
    if 'Unnamed: 0' in train_df.keys():
        train_df = train_df.drop('Unnamed: 0', axis=1)

    train_df['label_name'] = train_df['label']
    train_df['label'] = train_df['label_name'].replace({'neutral':0, 'entailment':1, 'contradiction':2})

    train_df['SEP_ind'] = train_df['context1'].apply(lambda x : len(tokenizer.tokenize(x))+1) # +1 is [CLS]
    train_df['context2'] = train_df['context2'].fillna(' ')
    train_df['context2'] = train_df['context2'].apply(lambda x : x.lower())

    train_df.to_csv('Dataset_infile\_MNLI.csv')

def create_SQuAD_df(file_path, tokenizer = None, data_size = 0):
    #處理好 dataframe
    df_train = pd.read_csv(file_path)
    if(data_size != 0):
        # df_train = df_train.sample(data_size)
        # df_train = df_train.reset_index(drop=True)
        df_train = df_train[0:data_size]
        df_train = df_train.sample(data_size).reset_index(drop=True)

    # question,context

    for i in range(len(df_train)):
        if( len(tokenizer.tokenize(df_train['question'][i])) + len(tokenizer.tokenize(df_train['context'][i])) >= 510 ):
            df_train = df_train.drop(i, axis=0)
            i = i-1
    
    df_train = df_train.reset_index(drop=True)
    
    df_train['answer_start'] = df_train['answer_start'].apply(lambda x : -1 if np.isnan(x) else int(x))
    df_train['text'] = df_train['text'].fillna(' ')
    df_train['text'] = df_train['text'].apply(lambda x : x.lower())
    df_train['label'] = df_train['answer_start'].apply(lambda x : 0 if x == -1 else 1)

    df_train['TKstart'] = pd.Series([0] * len(df_train))
    df_train['TKend'] = pd.Series([0] * len(df_train))

    for i in range(len(df_train)):
        df_train['TKstart'][i], df_train['TKend'][i] = count_the_TKbeg_and_TKend(df_train.iloc[i]['context'], df_train.iloc[i]['answer_start'], df_train.iloc[i]['text'], tokenizer)

    # if ('index' in df_train.keys()):
    #     df_train = df_train.drop('index', axis=1)
    if ('c_id' in df_train.keys()):
        df_train = df_train.drop('c_id', axis=1)
    if 'Unnamed: 0' in df_train.keys():
        df_train = df_train.drop('Unnamed: 0', axis=1)

    df_train['SEP_ind'] = df_train['context'].apply(lambda x : len(tokenizer.tokenize(x))+1)

    df_train.to_csv('Dataset_infile\_SQuAD.csv')
    
    return df_train

def create_QNLI_df(file_path, tokenizer = None, data_size = 0):
    df = pd.read_csv(file_path, index_col=[0])
    if(data_size != 0):
        df = df[0:data_size]

    for i in range(len(df)):
        print(i)
        if( len(tokenizer.tokenize(df['question'][i])) + len(tokenizer.tokenize(df['sentence'][i])) >= 510 ):
            df = df.drop(i, axis=0)
            i = i-1

    balanced_df = get_balanced_df(df, column_name='label')
    balanced_df = balanced_df.reset_index(drop=True)
    df = balanced_df

    df['label_name'] = df['label']
    df['label'] = df['label_name'].replace({'not_entailment':0, 'entailment':1})

    df['context1'] = df['question']
    df['context2'] = df['sentence']

    df = df.drop(['sentence','question'], axis=1)

    df['SEP_ind'] = df['context1'].apply(lambda x : len(tokenizer.tokenize(x))+1) # +1 is [CLS]
    df['context2'] = df['context2'].fillna(' ')

    df.to_csv('Dataset_infile\_QNLI.csv')

def create_RTE_df(file_path, tokenizer = None, data_size = 0):
    df = pd.read_csv(file_path, index_col=[0])
    if(data_size != 0):
        df = df[0:data_size]

    for i in range(len(df)):
        print(i)
        if( len(tokenizer.tokenize(df['context1'][i])) + len(tokenizer.tokenize(df['context2'][i])) >= 510 ):
            df = df.drop(i, axis=0)
            i = i-1
    
    balanced_df = get_balanced_df(df, column_name='label')
    balanced_df = balanced_df.reset_index(drop=True)
    df = balanced_df

    df['label_name'] = df['label']
    df['label'] = df['label_name'].replace({'not_entailment':0, 'entailment':1})

    df['SEP_ind'] = df['context1'].apply(lambda x : len(tokenizer.tokenize(x))+1) # +1 is [CLS]
    df['context2'] = df['context2'].fillna(' ')

    df.to_csv('Dataset_infile\_RTE.csv')

####------------------------------------------------------------------------

# get dataset
####------------------------------------------------------------------------
def get_CoLA_dataset(df_CoLA, tokenizer, num_labels):
    return Classification_dataset(df_CoLA, tokenizer, num_labels)

def get_MNLI_dataset(df_MNLI, tokenizer, num_labels):
    return Classification_dataset(df_MNLI, tokenizer, num_labels)

def get_SQuAD_dataset(df_SQuAD, tokenizer, num_labels = None): #num_labels hasn't use but for formalize
    return QAdataset(df_SQuAD, tokenizer=tokenizer, num_labels=num_labels)

def get_QNLI_dataset(df_QNLI, tokenizer, num_labels):
    return Classification_dataset(df_QNLI, tokenizer, num_labels)

def get_RTE_dataset(df_RTE, tokenizer, num_labels):
    return Classification_dataset(df_RTE, tokenizer, num_labels)

get_dataset_dict = {
                    'CoLA': get_CoLA_dataset, 
                    'MNLI' : get_MNLI_dataset, 
                    'SQuAD' : get_SQuAD_dataset,
                    'QNLI' : get_QNLI_dataset,
                    'RTE' : get_RTE_dataset}

def get_dataset(dataset_name, df, tokenizer = None, num_labels = None):
    return get_dataset_dict[dataset_name](df, tokenizer, num_labels)
####------------------------------------------------------------------------

# collate batch function
####------------------------------------------------------------------------
def QA_collate_batch(sample): #sample is List
    input_ids_batch = [s[0] for s in sample]
    mask_batch = [s[1] for s in sample]
    token_batch = [s[2] for s in sample]
    label_batch = [s[3] for s in sample]
    SEP_index_batch = [s[4] for s in sample]
    Start_batch = [s[5] for s in sample]
    End_batch = [s[6] for s in sample]

    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True)
    mask_batch = pad_sequence(mask_batch, batch_first=True)
    token_batch = pad_sequence(token_batch, batch_first=True)
    label_batch = torch.tensor(label_batch, dtype=torch.float)

    return input_ids_batch, mask_batch, token_batch, label_batch, SEP_index_batch, Start_batch, End_batch

def QA_evaluate_collate_batch(sample):
    input_ids_batch = [s[0] for s in sample]
    mask_batch = [s[1] for s in sample]
    token_batch = [s[2] for s in sample]
    label_batch = [s[3] for s in sample]
    SEP_index_batch = [s[4] for s in sample]
    index_batch = [s[5] for s in sample]

    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True)
    mask_batch = pad_sequence(mask_batch, batch_first=True)
    token_batch = pad_sequence(token_batch, batch_first=True)
    label_batch = torch.tensor(label_batch, dtype=torch.float)

    return input_ids_batch, mask_batch, token_batch, label_batch, SEP_index_batch, index_batch

def Classification_collate_batch(sample): #sample is List
    input_ids_batch = [s[0] for s in sample]
    mask_batch = [s[1] for s in sample]
    token_batch = [s[2] for s in sample]
    label_batch = [s[3] for s in sample]
    SEP_index_batch = [s[4] for s in sample]

    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True)
    mask_batch = pad_sequence(mask_batch, batch_first=True)
    token_batch = pad_sequence(token_batch, batch_first=True)
    label_batch = torch.tensor(label_batch, dtype=torch.float)

    return input_ids_batch, mask_batch, token_batch, label_batch, SEP_index_batch

####------------------------------------------------------------------------

#get dataloader
####------------------------------------------------------------------------
def get_CoLA_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Classification_collate_batch)

def get_MNLI_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Classification_collate_batch)

def get_SQuAD_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=QA_collate_batch)

def get_QNLI_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Classification_collate_batch)

def get_RTE_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Classification_collate_batch)

get_loader_dict = {
                    'CoLA': get_CoLA_dataloader, 
                    'MNLI' : get_MNLI_dataloader,
                    'SQuAD' : get_SQuAD_dataloader,
                    'QNLI' : get_QNLI_dataloader,
                    'RTE' : get_RTE_dataloader}

def get_dataloader(dataset_name, dataset, batch_size):
    dataset = get_loader_dict[dataset_name](dataset, batch_size)
    return dataset
####------------------------------------------------------------------------

class DataBox():
    def __init__(
    self,
    df_Data : DataFrame,
    test_size : int, 
    tokenizer : BertTokenizer,
    label_nums : int,
    batch_size : int,
    dataset_name : str
    ):
        df_train, df_test = train_test_split(df_Data, test_size=test_size, random_state=42, shuffle=True)
        self.df_train = df_train.reset_index(drop=True)
        self.df_test = df_test.reset_index(drop=True)
        self.label_nums = label_nums

        self.training_dataset = get_dataset(dataset_name, self.df_train, tokenizer, label_nums)
        self.training_dataloader = get_dataloader(dataset_name, self.training_dataset, batch_size)

        self.test_dataset = get_dataset(dataset_name, self.df_test, tokenizer, label_nums)
        self.test_dataloader = get_dataloader(dataset_name, self.test_dataset, batch_size)

        self.name = dataset_name

        self.H = {
        "train_loss": [],
        "train_acc": [],
        "test_loss":[],
        "test_acc": []
        }

####------------------------------------------------------------------------
class Classification_dataset(Dataset):
    def __init__(self, df, tokenizer, num_labels):
        self.df = df
        self.tokenizer = tokenizer
        self.num_labels = num_labels
    
    def __getitem__(self, index):
        df = self.df
        EC = self.tokenizer.encode_plus(df['context1'][index], df['context2'][index])
        
        input_ids = torch.tensor(EC['input_ids'])
        mask = torch.tensor(EC['attention_mask'])
        token = torch.tensor(EC['token_type_ids'])
        label = [0.] * self.num_labels
        label[df['label'][index]] = 1.

        return input_ids, mask, token, label, df['SEP_ind'][index]
    
    def __len__(self):
        return len(self.df)
    
class QAdataset(Dataset):
    def __init__(self, df, tokenizer, num_labels):
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.df = df
    
    def __getitem__(self, index):
        df = self.df
        EC = self.tokenizer.encode_plus(df['context'][index], df['question'][index])
        
        input_ids = torch.tensor(EC['input_ids'])
        mask = torch.tensor(EC['attention_mask'])
        token = torch.tensor(EC['token_type_ids'])
        label = [0.] * self.num_labels
        label[df['label'][index]] = 1.

        return input_ids, mask, token, label, df['SEP_ind'][index], df['TKstart'][index], df['TKend'][index]
    
    def __len__(self):
        return len(self.df)
    
#### this is use when do the evaluation
class QA_evalaute_dataset(Dataset):
    def __init__(self, df, tokenizer, num_labels):
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.df = df
    
    def __getitem__(self, index):
        df = self.df
        EC = self.tokenizer.encode_plus(df['context'][index], df['question'][index])
        
        input_ids = torch.tensor(EC['input_ids'])
        mask = torch.tensor(EC['attention_mask'])
        token = torch.tensor(EC['token_type_ids'])
        label = [0.] * self.num_labels
        label[df['label'][index]] = 1.

        return input_ids, mask, token, label, df['SEP_ind'][index], index
    
    def __len__(self):
        return len(self.df)
####------------------------------------------------------------------------
