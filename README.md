 
# Hate-Speech-Detection.
DeepLearning Project, masters in NLP, second year.


This project is for the binary classification of toxic comments using rcnn. We implemented using two different tokenization.
- Normal word tokenization
- word embedding with subword information (byte pair encoding using sentence piece)

# Outline

1. [Directory Structure](#directory-structure)
2. [Introduction](#introduction)
3. [Installation](#installation)
4. [Dataset](#dataset)
6. [Result](#Result)
7. [Contributing](#contributing)
8. [Licence](#licence)


## Directory structure

```
├── data
│   ├── all_train.csv ## train and val data merged together
│   ├── split_data.py ## to merge the main file into train, test and val, for development purpose only.
│   ├── test.csv ## test data
│   ├── train.csv ## train data
│   └── val.csv ## calidation data
├── inference_text.txt
├── output
│   ├── best.pt ## trained model 1
│   ├── best_spm.pt ## trained model 2(using sentence piece)
│   ├── non_letters.pkl ## saved non letters and special characters
│   ├── sample.csv # for spm, only tweets saved
│   ├── spm_user.model ## sentence piece model trained
│   ├── spm_user.vocab ## sentence piece model vocab
│   ├── word_to_index.pickle # word to index for model 1
│   └── word_to_index_spm.pickle # word to index for model 2
├── README.md 
├── requirements.txt # requirements
├── src ## main source codes
│   ├── dataset.py ## dataset classes
│   ├── evaluate.py # evalaution code
│   ├── main.py ## main file to be called
│   ├── params.py ## params defined
│   ├── rcnn.py ## model class
│   └── train.py ## training file
└── tree.txt ## directory structure

```



## Introduction

This project is the implementation for binary classification. The steps are cleaning the dataset and removing non letters, then creating dataloader, training, evaluating and saving the model.
The training and validation losses and accuracy can be tracked: [model1](https://wandb.ai/sarmila433/Toxic%20comment%20classification)
[model2](https://wandb.ai/sarmila433/Toxic%20comment%20classification)


## Installation


In order to get the model to run, follow these installation instructions.


<!-- ### Requirements -->
Pre-requisites:

    python>=3


### 1. Clone the repository

    git clone [link]

_Optional_: use the package manager [pip](https://pip.pypa.io/en/stable/) to install a vitual environment.

    bash
    pip install virtualenv
    
    
    
#### 2. Navigate to the folder with the cloned git repository

#### 3. Create Virtual Environment

    virtualenv <name of env> --python /usr/bin/python[version] or <path to your python if its not the mentioned one>
    
Conda:

    conda create --name <name of your env> python=3.7

#### 4. Activate Virtual Environment

    source name_of_env/bin/activate
On Windows:

    name_of_env\Scripts\activate
Conda:

    conda activate <name of your env>

(To leave the virtual environment, simply run: ```deactivate``` (with virtualenv) or ```conda deactivate``` (with Conda))

---

### 5. Install Requirements

    pip install -r requirements.txt
        
Conda:

    conda install pip
    pip install -r requirements.txt


---


#### 6. You also need to download the spacy model:

    python -m spacy download en_core_web_sm

---

### 7. Initial Downloads
you can directly use the model for either test or inference. Since we have two models, go to this [link](), download the zip file then unzip the file inside output/ directory



If you want to train first, then skip this section!
************************************************************************************************************************************
**_YAY!!_** Installation is done! Now you can jump to the execution part and run the web app.


## Execution

src/main.py needs to be run. Detail about the file arguments

```
usage: main.py [-h] [--type t] [--textfile TEXTFILE] [--sentencepiece SENTENCEPIECE]

train or text the toxicity classification

optional arguments:
  -h, --help            show this help message and exit
  --type t              it could be either train, test or inference
  --textfile TEXTFILE   text file to test the toxicity
  --sentencepiece SENTENCEPIECE
                        to activate sentence piece dataloader
```

### train
#### using word tokenization + embeddings of words

```
python3 src/main.py --type train
```
*** model will be saved inside output/ directory ***

### using sentence piece + embeddings of subwords 

```
python3 src/main.py --type train --sentencepiece True
```
*** model will be saved inside output/ directory ***

### test

#### using word tokenization + embeddings of words

```
python3 src/main.py --type test
```
*** model will be saved inside output/ directory ***

### using sentence piece + embeddings of subwords 

```
python3 src/main.py --type test --sentencepiece True
```
*** model will be saved inside output/ directory ***

### inference

#### using word tokenization + embeddings of words
```
python3 src/main.py --textfile <filepath> --type test
```

### using sentence piece + embeddings of subwords 
```
python3 src/main.py --type test --textfile <filepath> --sentencepiece True
```
---


## Dataset

 [Link](https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech)
We splitted the data as 60:20:20. 60 percent for training, 20 for test and rest for validation. Data is splitted and saved inside data/ directory. We highly reccomend to use splitted data as the sentence piece vocab is set according to the train and validation data. 
**Only if you want to re split, then simply put the train.csv file inside data/ and use the split_data python file to create the split. Then, the sentencepiece creation should have vocab size according to these newly splitted data. When you are training with sentence piece if you get error about vocab size, then use the suggested vocab size on main.py file. Similarly, you need to change the params for padding_idx_spm as vocab_size+1 and vocab_size_spm as vocab size of sentence piece.**

## Result
Result on test dataset:

### Model with word tokenization and data preprocessing

```
Metrics | #F1score | #Accuracy | #Precision | #Recall 
--- | --- | --- | --- |---
Values | 0.7798 | 95.6 | 0.8869 | 0.7234
```


### Model with sentence piece

```
Metrics | #F1score | #Accuracy | #Precision | #Recall 
--- | --- | --- | --- |--- 
Values | 301 | 283 | 290 | 286
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Licence
[MIT](https://choosealicense.com/licenses/mit/)
