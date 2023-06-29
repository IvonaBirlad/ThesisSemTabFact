# dirpath = "~/models"
# print(dirpath)

from pathlib import Path
import pandas as pd
import os
from sklearn.utils import shuffle

ROOT_DIRECTORY_PATH = str(Path(__file__).parent)
PNG_PATH_MAN = "png_data/data_aug.csv"
PNG_PATH_AUTO ="png_data_auto/data_aug_auto.csv"
# print(os.path.join(ROOT_DIRECTORY_PATH, PNG_PATH))
#
# #combine data from manual and automatic annotations
# data_man = pd.read_csv(os.path.join(ROOT_DIRECTORY_PATH, PNG_PATH_MAN), index_col=0)
# data_auto = pd.read_csv(os.path.join(ROOT_DIRECTORY_PATH, PNG_PATH_AUTO), index_col=0)
# data_auto = shuffle(data_auto, random_state=42, n_samples=3872)
# data = pd.concat([data_man, data_auto])
# data = shuffle(data, random_state=42)
# data.reset_index(inplace=True, drop=True)
#
# data.to_csv("data_aug_10000.csv", index=False)

data_read = pd.read_csv("data_aug_10000.csv")

# print(data_read.head(5))


# print(data.shape)
data_read["table_name"] = ROOT_DIRECTORY_PATH + "/" + data_read["table_name"]
from sklearn.model_selection import train_test_split as tts

train_df, valid_df = tts(data_read, random_state=42, stratify=data_read['label'], shuffle=True)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
print(train_df.head())
