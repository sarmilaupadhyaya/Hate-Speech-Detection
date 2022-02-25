import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
train_df, test_df = train_test_split(df, test_size=0.2)
train_df.to_csv("train_val.csv")
train_df, val_df = train_test_split(train_df, test_size=0.2)

train_df.to_csv("train.csv")
test_df.to_csv("test.csv")
val_df.to_csv("val.csv")

