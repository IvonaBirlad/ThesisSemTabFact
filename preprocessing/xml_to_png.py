# Create the main df with table_name and statement as columns as well as the PNG files of the tables


import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET
import dataframe_image as dfi

# Set to True if you want to include table metadata (caption, legend, footer)
INCLUDE_TABLE_META = True

# Global variables
table_names_list = []
statement_list = []
labels = []


def getTableDims(t):
    rows = t.findall("row")  # can be used for making the table
    n_rows = len(rows)  # Total number of rows in the table, including sub-headers etc
    # Now we need to find n_cols
    max_col = 0
    # Iterate over all cells in the table
    for c in t.iter("cell"):
        col_end = int(c.attrib["col-end"])
        if col_end > max_col:
            max_col = col_end
    n_cols = max_col + 1  # Since 0 based indexing
    return n_rows, n_cols


def xmlToDataframe(t):
    n_rows, n_cols = getTableDims(t)

    # Initialize a n_rows x n_cols table
    unk = "[UNK]"  # Use whatever unknown token you want here
    table = np.array([[unk] * n_cols] * n_rows, dtype="object")  # Convert to numpy array for easier slicing and setting

    for c in t.iter("cell"):
        col_start = int(c.attrib["col-start"])
        col_end = int(c.attrib["col-end"])
        row_start = int(c.attrib["row-start"])
        row_end = int(c.attrib["row-end"])
        ctext = c.attrib["text"].replace(",", " ")  # Replace commas else there will be errors while reading the CSV
        table[row_start:row_end + 1,
        col_start:col_end + 1] = ctext  # Set entire submatrix to the text present in the cell

    if INCLUDE_TABLE_META:
        # Now, add the table legend, caption and footer
        # Adding legend before caption ensures that caption can be easily added on top of the legend
        legend = t.find("legend")
        if legend != None:
            legend = legend.attrib["text"].replace(",", " ")
            # Add an extra row at the top of the table and fill all cells with the legend
            row_legend = np.array([""] * n_cols, dtype="object")
            row_legend[0] = legend
            table = np.vstack((row_legend, table))

        caption = t.find("caption")
        if caption != None:
            caption = caption.attrib["text"].replace(",", " ")
            # Add an extra row at the top of the table (could be above the added legend) and fill all cells with the caption
            row_caption = np.array([""] * n_cols, dtype="object")
            row_caption[0] = caption
            table = np.vstack((row_caption, table))

        footnote = t.find("footnote")
        if footnote != None:
            footnote = footnote.attrib["text"].replace(",", " ")
            row_footnote = np.array([""] * n_cols, dtype="object")
            row_footnote[0] = footnote
            table = np.vstack((table, row_footnote))
            # Add an extra row at the bottom of the table and fill all cells with the footnote

    # Row and column headers are default kept as 0,1,2,3..., this may create problems if TAPAS extracts this seperately via df.columns
    df_table = pd.DataFrame(data=table, index=None, columns=None)
    return df_table


def pngBuilder(base_path, png_path, filename):
    outpath = f"{base_path}/{filename}"
    filename_raw = filename.split(".")[0]  # Remove the .xml extension

    png_rel_path = png_path.split("/")[-1]

    allowed_types = ["entailed", "refuted", "unknown"]
    label_map = {"entailed": 1, "refuted": 0, "unknown": 2}

    tree = ET.parse(outpath)  # The element tree
    root = tree.getroot()  # Root node

    i = 1  # Index for the table
    for t in root.iter("table"):  # Iterate over all tables

        # Create and save the table as a PNG
        df_table = xmlToDataframe(t)  # Convert to dataframe
        tmp_outpath = f"{filename_raw}_t{i}.png"
        df_table.dfi.export(f"{png_path}/{tmp_outpath}", max_rows=-1)

        # Build the PNG
        for s in t.iter("statement"):
            stype = s.attrib["type"]
            stext = s.attrib["text"].replace(",", " ")
            if stype in allowed_types:
                label = label_map[stype]
                labels.append(label)
                table_names_list.append(f"{png_rel_path}/{tmp_outpath}")
                statement_list.append(stext)
        i += 1


if __name__ == "__main__":

    BASE_PATH = './xml/output_test/'
    PNG_PATH = './png_data_test'
    DF_SAVE_PATH = "./png_data_test/data_test.csv"
    FILE_RE = "*.xml"  # RegEx used while scanning the input directory

    if not os.path.exists(f"{PNG_PATH}"):
        os.makedirs(f"{PNG_PATH}")

    for fullpath in tqdm(sorted(glob.glob(os.path.join(BASE_PATH, FILE_RE)))):
        filename = os.path.basename(os.path.normpath(fullpath))
        pngBuilder(BASE_PATH, PNG_PATH, filename)

    df = pd.DataFrame.from_dict({
        "table_name": table_names_list,
        "statement": statement_list,
        "label": labels
    })

    df.index.name = "id"
    df.to_csv(DF_SAVE_PATH)
