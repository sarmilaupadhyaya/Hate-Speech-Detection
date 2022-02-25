import pickle
import string
from collections import Counter
import spacy
nlp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
import torch
from torchtext.vocab import Vocab
from torch.utils.data import Dataset, DataLoader
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_tokenizer, sentencepiece_numericalizer
from torchtext.data.functional import generate_sp_model
import sentencepiece as sp
from params import *


def inference_tokens_spm(sentence):
    """
    returns integers for the tokens in the sentence using sentence piece
    Params:
    sentence (str): input sentence
    Returns:
    tensor

    """
    # load the sentence piece
    tokenizer = load_sp_model(sentencepiece)
    sp_tokens_generator=sentencepiece_tokenizer(sp_model=tokenizer)
    sp_id_generator= sentencepiece_numericalizer(tokenizer)

    with open(w2idx_path_spm,"rb") as f:
        word_to_index = pickle.load(f)

    tokens = list(sp_id_generator([sentence]))[0]

    for i in range(max_seq_len - len(tokens)):
        tokens.append(word_to_index["[PAD]"])
    return torch.LongTensor(tokens[:max_seq_len])


def inference_tokens(sentence):
    """
    returns integers for the tokens in the sentence using sentence piece
    Params:
    sentence (str): input sentence
    Returns:
    tensor

    """
    with open(w2idx_path,"rb") as f:
        word_to_index = pickle.load(f)
    with open(non_letters_path,"rb") as f:
        non_letters = pickle.load(f)
    def remove_stuff(sentence):
            for i in non_letters:
                sentence = sentence.lower().replace(i, "")
            return(sentence)

    text = sentence.lower()
    tokens = []
    text = remove_stuff(text)
    text_parsed = nlp(text)
    for i in text_parsed:
        if not i.text.isspace():
            tokens.append(i.lemma_)
    for i in range(128 - len(tokens)):
        tokens.append('[PAD]')
    for i,token in enumerate(tokens):
        if token not in word_to_index:
            tokens[i] = "UNK"
    return torch.LongTensor([word_to_index[word] for word in tokens])

def create_non_letters(df):
    """
     gets the special symbol
     params:

     df (pandas.dataframe): train dataframe

     returns:
     list of non characters

    """
    vocab = []
    all_text = df["tweet"].str.lower().tolist()
    letters = string.ascii_lowercase
    word_string = ' '.join(all_text)
    not_letters = set([char for char in word_string if char not in letters and char != ' '])
    #save non letters
    with open(non_letters_path, "wb") as f:
        pickle.dump(not_letters,f)
    print("non letters created and dumped successfully !!")

class TextDataset(Dataset):

    def __init__(self, df, max_seq_length=128):
        # create variables for storing the attributes of the class (text_dir, max_seq_length, and the list of text files)
        self.max_seq_len = max_seq_length
        self.df = df 
        # extract the labels from the given texts using _get_labels()
        self.labels = self._get_labels()

        #create a dictionary <word_to_index> which takes words as keys and unique integers (0, 1, 2, ...) as values
        with open(w2idx_path,"rb") as f:
            self.word_to_index = pickle.load(f)


        # special token [PAD] used for padding text to a fixed length (check _preprocess_text() for details)
        self.word_to_index['[PAD]'] = len(self.word_to_index)
        self.word_to_index["UNK"] = len(self.word_to_index)
        with open(w2idx_path,"wb") as f:
            pickle.dump(self.word_to_index, f)

    def __len__(self):
        # return the length of the list of text files
        return len(self.df)

    def __getitem__(self, index):
        # get the index-th text file from the list of text files defined in __init__
        text = self.df["tweet"].tolist()[index]

        # return a list of all tokens in the text and the respective label (use the _tokenize_text method)
        tokens = self._tokenize_text(text)

        # use the word_to_index mapping to transform the tokens into indices and save them into an IntTensor
        x = torch.LongTensor([self.word_to_index[word] for word in tokens])

        # get the index-th label and store it into a FloatTensor
        y = torch.LongTensor([self.labels[index]])

        # stores the text indices and the label into a dictionary
        features = {'token_ids': x, 'labels': y}
        return features

    def _find_files(self, directory, pattern='*.txt'):
        """Recursively finds all files matching the pattern."""
        files = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
        return files

    def _get_labels(self):
        """Extracts the labels from the given text files."""
        labels = []
        labels = self.df["label"].astype(int).tolist()
        return labels

    def _create_vocabulary(self):
        """Creates a vocabulary of unique words from the given text files."""
        all_texts = [list(open(filepath, 'r'))[0].strip().lower() for filepath in self.text_files]
        letters = string.ascii_lowercase
        word_string = ' '.join(all_texts)
        not_letters = set([char for char in word_string if char not in letters and char != ' '])
        for char in not_letters:
            word_string = word_string.replace(char, " ")
        vocab = set(word_string.split())
        return list(vocab)
    
    def _tokenize_text(self, sentence):
        """
        Removes non-characters from the text and pads the text to max_seq_len.
        *!* Padding is necessary for ensuring that all text_files have the same size
        *!* This is required since DataLoader cannot handle tensors of variable length

        Returns a list of all tokens in the text
        """
        def create_non_letters():
            with open(non_letters_path, "rb") as f:
                non_letters = pickle.load(f)
            return non_letters
        
        def remove_stuff(sentence):
            for i in not_letters:
                sentence = sentence.lower().replace(i, "")
            return(sentence)
        
        text = sentence.lower()
        tokens = []
        not_letters = create_non_letters()
        text = remove_stuff(text)
        text_parsed = nlp(text)
        for i in text_parsed:
            if not i.text.isspace():
                tokens.append(i.lemma_)
        for i in range(128 - len(tokens)):
            tokens.append('[PAD]')
        for i,token in enumerate(tokens):
            if token not in self.word_to_index:
                tokens[i] = "UNK"
        return tokens


class TextDatasetSpm(Dataset):

    """
    dataset class to return byte pair tokens using sentence piece model

    """

    def __init__(self, df,  sp_model = "output/spm_user.model",max_seq_length=119):
        # create variables for storing the attributes of the class (text_dir, max_seq_length, and the list of text files)
        self.max_seq_len = max_seq_length
        self.df = df 
        # extract the labels from the given texts using _get_labels()
        self.labels = self._get_labels()
        
      
        self.tokenizer = load_sp_model(sp_model)
        self.sp_tokens_generator=sentencepiece_tokenizer(sp_model=self.tokenizer)
        self.sp_id_generator= sentencepiece_numericalizer(self.tokenizer)
        
        print("loading sentence piece model and vocab")
        
        self.sp = sp.SentencePieceProcessor()
        self.sp.load(sp_model)
        print("loading sentence piece vocab processor")
        
        self.word_to_index = {self.sp.id_to_piece(id): id for id in range(self.sp.get_piece_size())}
        # special token [PAD] used for padding text to a fixed length (check _preprocess_text() for details)
        self.word_to_index['[PAD]'] = len(self.word_to_index)
        with open(w2idx_path_spm,"wb") as f:
            pickle.dump(self.word_to_index, f)

        
    def __len__(self):
        # return the length of the list of text files
        return len(self.df)

    def __getitem__(self, index):
        # get the index-th text file from the list of text files defined in __init__
        
        text = self.df["tweet"].tolist()[index]

        # return a list of all tokens in the text and the respective label (use the _tokenize_text method)
        tokens = self._tokenize_text(text)
        
        x = torch.LongTensor(tokens)

        # get the index-th label and store it into a FloatTensor
        y = torch.LongTensor([self.labels[index]])

        # stores the text indices and the label into a dictionary
        features = {'token_ids': x, 'labels': y}
        return features


    def _get_labels(self):
        """Extracts the labels from the given text files."""
        labels = []
        labels = self.df["label"].astype(int).tolist()
        return labels

    
    def _tokenize_text(self, text):
        """
        *!* Padding is necessary for ensuring that all text_files have the same size
        *!* This is required since DataLoader cannot handle tensors of variable length

        Returns a list of all tokens in the text
        """
        text = text.lower()
        letters = string.ascii_lowercase
        not_letters = set([char_ for char_ in text if char_ not in letters and char_ != ' '])
        for char in not_letters:
            text = text.replace(char, " ")
        
        tokens = list(self.sp_id_generator([text]))[0]
         
        for i in range(self.max_seq_len - len(tokens)):
            
            tokens.append(self.word_to_index["[PAD]"])
        return tokens[:self.max_seq_len]



