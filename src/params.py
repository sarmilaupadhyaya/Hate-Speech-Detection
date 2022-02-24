

train_df="data/train.csv"
test_df = "data/test.csv"
val_df="data/val.csv"
all_train="data/all_train.csv"
model_path="output/best.pt"
model_path_spm="output/best_spm.pt"
w2idx_path = "output/word_to_index.pickle"
w2idx_path_spm="output/word_to_index_spm.pickle"
sentencepiece="output/spm_user.model"
non_letters_path="output/non_letters.pkl"

## model parameters
batch_size=64
vocab_size=37633
embedding_dim=300
filters=300
hidden_size=512
hidden_size_linear=512
class_num=2
dropout=0.0
epochs=2
padding_idx=37631

vocab_size_spm=23457
padding_idx_spm=23456


