import sys
import argparse
import torch
import csv
import pandas as pd
from torchtext.data.functional import generate_sp_model
import params
from rcnn import RCNN
from train import *
from dataset import *

##data path
train_df_path = params.train_df
test_df_path = params.test_df
val_df_path = params.val_df

def train_sentencepiece(df):
    """
    function to train sentence piece on training and validation data
    """
    df = pd.read_csv(df)
    with open('./sample.csv', 'w', newline='', encoding='utf-8') as f:
        for x in df["tweet"].tolist():
            f.write(x)
            f.write("\n")
    #train and it will save a model bames spm_use.model
    generate_sp_model('./sample.csv',vocab_size=23456, model_prefix='output/spm_user')

def fetch_inference_tokens(text, sentencepiece):
    """
    gets the token ids for single sentence using wither sentencepiece implemented textdataset or simple tokenization

    """

    if sentencepiece:
        x = inference_tokens_spm(text)
    else:
        x = inference_tokens(text)
    return x

def get_dataloader(type_action, sentencepiece):

    """
    loads dataloader for train, test and validation dataset

    """

    if type_action == "train":
        # reads datafram
        train_df = pd.read_csv(train_df_path)
        val_df = pd.read_csv(val_df_path)
        test_df = pd.read_csv(test_df_path)
        # need to call two different type of dataset depending on the type of tokenization
        if sentencepiece:
            train_sentencepiece(params.all_train)
            train_dataset = TextDatasetSpm(train_df)
            validation_dataset = TextDatasetSpm(val_df)
        else:
            create_non_letters(pd.concat([train_df, val_df,test_df], ignore_index=True, sort=False))
            train_dataset = TextDataset(train_df)
            validation_dataset = TextDataset(val_df)

        # created dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
        val_dataloader = DataLoader(validation_dataset, batch_size=params.batch_size, shuffle=True)
        return train_dataloader, val_dataloader

    # if test then sends dataloader of test only
    elif type_action == "test":
        test_df = pd.read_csv(test_df_path)
        if sentencepiece:
            test_dataset = TextDatasetSpm(test_df)
        else:
            test_dataset = TextDataset(test_df)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        return test_dataloader



def main(type_action, filepath,sentencepiece=False):
    """
    function to train, test and validate. Also works for inference
    params:
    type_actions (str): either train, test or inference
    filepath (str): files containing text for inference

    """

    
    if  sentencepiece:
        vocab_size=params.vocab_size_spm
        padding_idx=params.padding_idx_spm
        model_path=params.model_path_spm
    else:
        vocab_size=params.vocab_size
        padding_idx=params.padding_idx
        model_path=params.model_path
    model = RCNN(vocab_size=vocab_size,
                 embedding_dim=params.embedding_dim,
                 hidden_size=params.hidden_size,
                 hidden_size_linear=params.hidden_size_linear,
                 class_num=params.class_num,
                 dropout=params.dropout,padding_idx=padding_idx)

    if type_action == "train":

        train_dataloader, val_dataloader = get_dataloader(type_action,sentencepiece)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        train_(model, optimizer, train_dataloader, val_dataloader,model_path,epochs=params.epochs)
        print('******************** Train Finished ********************')

    elif type_action == "test":
        
        test_dataloader = get_dataloader(type_action,sentencepiece)
        model.load_state_dict(torch.load(model_path))
        _ , accuracy, precision, recall, f1, cm = evaluate(model, test_dataloader)
        print('-'*50)
        print(f'|* TEST SET *| |ACC| {accuracy:>.4f} |PRECISION| {precision:>.4f} |RECALL| {recall:>.4f} |F1| {f1:>.4f}')
        print('-'*50)
        print('---------------- CONFUSION MATRIX ----------------')
        for i in range(len(cm)):
            print(cm[i])
        print('--------------------------------------------------')
    elif type_action == "inference":
        if filepath == "":
            sys.exit("Please give the input as a text file")
        texts  = open(filepath, "r").readlines()
        print(texts)
        for text in texts:
            X = fetch_inference_tokens(text,sentencepiece)
            model.load_state_dict(torch.load(model_path))
            print(torch.unsqueeze(X,0).shape)
            result = model(torch.unsqueeze(X,0))
            print(result)




if __name__=='__main__':

    ## parse all the documents
    ## get args about training or only testing or inference, if inference text 
    parser = argparse.ArgumentParser(description='train or text the toxicity classification')
    parser.add_argument('--type', metavar='t', type=str,
                    help='it could be either train, test or inference')
    parser.add_argument('--textfile', default="",type=str,
                    help='text file to test the toxicity')
    parser.add_argument('--sentencepiece',type=bool, default=False,
                    help='to activate sentence piece dataloader')

    args = parser.parse_args()
    type_action = args.type
    filepath = args.textfile
    sentence_piece=args.sentencepiece
    main(type_action, filepath, sentence_piece)

