 
# Hate-Speech-Detection.
DeepLearning Project, masters in NLP, second year.


This project is for the binary classification of toxic comments using rcnn. We implemented using two different tokenization.
- Normal word tokenization
- byte pair encoding using sentence piece

# Outline

1. [Directory Structure](#directory-structure)
2. [Introduction](#introduction)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Result](#Result)
6. [Contributing](#contributing)
7. [Licence](#licence)


## Directory structure

    ```
    
    ```


## Introduction

xxx


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

---


## Dataset

<!-- - [](https://paperswithcode.com/datasets) -->


## Results
Result on test dataset:

### Model with word tokenization

### Model with sentence piece

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Licence
[MIT](https://choosealicense.com/licenses/mit/)
