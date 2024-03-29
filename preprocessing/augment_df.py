# Augment the DF directly and inserts unknown statements in each table from other tables in the same file


import re
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

random.seed(0) # Set seed

csv_path = "./png_data_auto/data_manual_auto.csv"
save_path = "./png_data_auto/data_aug_auto.csv"

df = pd.read_csv(csv_path, index_col = "id")

# The DF which will contain only the unknown statements, will be concatenated with df later 
df_unk = {
    "table_name" : [],
    "statement" : [],
    "label" : []
}

n_f1 = 0 # No. of files with only a single table

table_names = df["table_name"].unique()
n_tables = len(table_names)

# We add unknown statements for each table
for i, tname_1 in enumerate(tqdm(table_names)):

    random.seed(i) # Different seed for each table
    print(tname_1)
    # Get no. of unk statements to insert
    mask_1 = (df["table_name"] == tname_1)
    labels = df["label"][mask_1] # Contains all labels for table tname_1
    n_ent = np.sum(labels == 1)
    n_ref = np.sum(labels == 0)

    # Formula used for determining how many unknown statements to insert for a table
    n_unk = max(min(n_ent, n_ref), 1)

    # Get all tables in the same file as tname_1, tname_1 of the form: path/filename_tableno.csv
    root_path = tname_1.split("/")[0]
    # print(root_path)
    fname   = tname_1.split("/")[-1].split("_")[0]
    # print(fname)
    pattern = re.compile(f"{root_path}/{fname}_a[0-9]+.png")
    # print(tname_1)
    matches = list(filter(pattern.match, table_names))
    matches.remove(tname_1) # Remove self, since that will be always matched
    n_matches = len(matches)

    if n_matches > 0: # i.e atleast 1 other table in the same file
        fstmts = [] # All statements in the same file
        for tname_2 in matches:
            mask_2 = (df["table_name"] == tname_2) # All statements for table tname_2
            tstmts = df["statement"][mask_2].to_list()
            fstmts += tstmts
        n_potential = len(fstmts) # List of potential unknown statements
        if n_potential > n_unk:
            unknowns = random.sample(fstmts, k = n_unk)
        else:
            unknowns = fstmts.copy()

    else:
        # File has only 1 table, add unknown statements from other files
        mask_3 = (df["table_name"] != tname_1) # All statements apart from statements in table_1
        others = df["statement"][mask_3].to_list()
        n_potential = len(others)
        if n_potential > n_unk:
            unknowns = random.sample(others, k = n_unk)
        else:
            unknowns = potential_stmts.copy()
        n_f1 += 1

    # Append unk statements to the df
    for statement in unknowns:
        df_unk["table_name"].append(tname_1)
        df_unk["statement"].append(statement)
        df_unk["label"].append(2)

df_unk = pd.DataFrame.from_dict(df_unk)
df_unk.index.name = "id"

df_aug = pd.concat([df, df_unk], ignore_index = True)
df_aug.index.name = "id"

print(f"There are {n_tables} tables in the data")
print(f"{n_f1} files contain a single table only")
print(f"Class Distribution after augmentation:")
print(df_aug["label"].value_counts())

df_aug.to_csv(save_path)