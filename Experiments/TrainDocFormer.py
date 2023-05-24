## A small Introduction about the Model:

### Installing the Libraries

# Installing the dependencies (might take some time)

## Cloning the repository
# !git clone https: // github.com / uakarsh / docformer.git

## Logging into wandb


import wandb
wandb.login(key="X") #Add your wandb key

## 2. Libraries

## Importing the libraries

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import torchvision.models as models

## Adding the path of docformer to system path
import sys

sys.path.append('./docformer/src/docformer/')

## Importing the functions from the DocFormer Repo
from dataset import create_features
from modeling import DocFormerEncoder, ResNetFeatureExtractor, DocFormerEmbeddings, LanguageFeatureExtractor
from transformers import BertTokenizerFast

## Hyperparameters

seed = 42
target_size = (500, 384)
n_classes = 3

## Setting some hyperparameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## One can change this configuration and try out new combination
config = {
    "coordinate_size": 96,  ## (768/8), 8 for each of the 8 coordinates of x, y
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "image_feature_pool_shape": [7, 7, 256],
    "intermediate_ff_size_factor": 4,
    "max_2d_position_embeddings": 1024,
    "max_position_embeddings": 128,
    "max_relative_positions": 8,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "shape_size": 96,
    "vocab_size": 30522,
    "layer_norm_eps": 1e-12,
}

## Importing the data
from pathlib import Path
import os
ROOT_DIRECTORY_PATH = str(Path(__file__).parent.parent)
PNG_PATH = "png_data/data_aug.csv"
print(os.path.join(ROOT_DIRECTORY_PATH, PNG_PATH))

data = pd.read_csv(os.path.join(ROOT_DIRECTORY_PATH, PNG_PATH), index_col=0)
data["table_name"] = ROOT_DIRECTORY_PATH + "/" + data["table_name"]

from sklearn.model_selection import train_test_split as tts

train_df, valid_df = tts(data, random_state=seed, stratify=data['label'], shuffle=True)

train_df = train_df.reset_index().drop(columns=['id'], axis=1)
valid_df = valid_df.reset_index().drop(columns=['id'], axis=1)


## 3. Making the dataset

## Creating the dataset

class SemTabFactData(Dataset):

    def __init__(self, image_list, label_list, statement_list, target_size, tokenizer, max_len=512, transform=None):

        self.image_list = image_list
        self.label_list = label_list
        self.statement_list = statement_list
        self.target_size = target_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.label_list[idx]
        statement = self.statement_list[idx]

        ## More on this, in the repo mentioned previously
        final_encoding = create_features(
            img_path,
            self.tokenizer,
            add_batch_dim=False,
            target_size=self.target_size,
            max_seq_length=self.max_len,
            path_to_save=None,
            save_to_disk=False,
            apply_mask_for_mlm=False,
            extras_for_debugging=False,
            use_ocr=True  # Please provide the bounding box and words or pass the argument "use_ocr" = True
        )
        if self.transform is not None:
            ## Note that, ToTensor is already applied on the image
            final_encoding['resized_scaled_img'] = self.transform(final_encoding['resized_scaled_img'])

        keys_to_reshape = ['x_features', 'y_features', 'resized_and_aligned_bounding_boxes']
        for key in keys_to_reshape:
            final_encoding[key] = final_encoding[key][:self.max_len]

        final_encoding['label'] = torch.as_tensor(label).long()

        statement_encoding = tokenizer(statement,
                                       padding="max_length",
                                       max_length=self.max_len,
                                       truncation=True)

        final_encoding['statement'] = torch.as_tensor(statement_encoding["input_ids"])
        return final_encoding


## Defining the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

from torchvision import transforms

## Normalization to these mean and std (I have seen some tutorials used this, and also in image reconstruction, so used it)
transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

train_ds = SemTabFactData(train_df['table_name'].tolist(), train_df['label'].tolist(), train_df['statement'].tolist(),
                          target_size, tokenizer, config['max_position_embeddings'], transform)
val_ds = SemTabFactData(valid_df['table_name'].tolist(), valid_df['label'].tolist(), valid_df['statement'].tolist(),
                        target_size, tokenizer, config['max_position_embeddings'], transform)

for key in list(train_ds[1].keys()):
    print_statement = '{0: <50}'.format(str(key) + " has a shape:")
    print(print_statement, train_ds[0][key].shape)

from PIL import Image, ImageDraw

try:
    original_image = Image.open(train_df["table_name"][5]).convert("RGB")
except:
    original_image = Image.new(mode="RGB", size=((500, 500)), color=(255, 255, 255))

sample = np.array(original_image)

# sample = np.transpose(sample, (1, 2, 0)).astype(np.uint8)
sample = Image.fromarray((sample * 255).astype(np.uint8))

# Visualizing the resized image
target_size = (512, 384)
resize_img = sample.resize(target_size)
resize_img

### Collate Function:

# from [here](https: // stackoverflow.com / questions / 65279115 / how - to - use - collate - fn - with-dataloaders)
#

def collate_fn(data_bunch):
    '''
    A function for the dataloader to return a batch dict of given keys

    data_bunch: List of dictionary
    '''

    dict_data_bunch = {}

    for i in data_bunch:
        for (key, value) in i.items():
            if key not in dict_data_bunch:
                dict_data_bunch[key] = []
            dict_data_bunch[key].append(value)

    for key in list(dict_data_bunch.keys()):
        dict_data_bunch[key] = torch.stack(dict_data_bunch[key], axis=0)

    return dict_data_bunch


## 4. Defining the DataModule

import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):

    def __init__(self, train_dataset, val_dataset, batch_size=4):
        super(DataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=collate_fn, shuffle=False)


datamodule = DataModule(train_ds, val_ds)


## 5. Modeling Part

class DocFormerForClassification(nn.Module):

    def __init__(self, config):
        super(DocFormerForClassification, self).__init__()

        self.resnet = ResNetFeatureExtractor(hidden_dim=config['max_position_embeddings'])
        self.embeddings = DocFormerEmbeddings(config)
        self.lang_emb = LanguageFeatureExtractor()
        self.config = config
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.linear_layer = nn.Linear(in_features=config['hidden_size'], out_features=n_classes)  ## Number of Classes
        self.encoder = DocFormerEncoder(config)
        self.linear = nn.Linear(256, 128)

    def forward(self, batch_dict):
        x_feat = batch_dict['x_features']
        y_feat = batch_dict['y_features']

        table_token = batch_dict['input_ids']
        statement_token = batch_dict['statement']
        img = batch_dict['resized_scaled_img']
        v_bar_s, t_bar_s = self.embeddings(x_feat, y_feat)
        v_bar = self.resnet(img)
        table_emb = self.lang_emb(table_token)
        statement_emb = self.lang_emb(statement_token)
        t_bar = table_emb + statement_emb

        out = self.encoder(t_bar, v_bar, t_bar_s, v_bar_s)
        out = self.linear_layer(out)
        out = out[:, 0, :]
        return out


## Defining pytorch lightning model
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchmetrics


class DocFormer(pl.LightningModule):

    def __init__(self, config, lr=5e-5):
        super(DocFormer, self).__init__()

        self.save_hyperparameters()
        self.config = config
        self.docformer = DocFormerForClassification(config)

        self.num_classes = n_classes
        self.train_accuracy_metric = torchmetrics.Accuracy(task='multiclass',
                                                           num_classes=self.num_classes)
        self.val_accuracy_metric = torchmetrics.Accuracy(task='multiclass',
                                                         num_classes=self.num_classes)
        self.f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", task='multiclass', num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", task='multiclass', num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro", task='multiclass',
                                                             num_classes=self.num_classes)
        self.recall_micro_metric = torchmetrics.Recall(average="micro", task='multiclass', num_classes=self.num_classes)
        self.label = []
        self.logit = []

    def forward(self, batch_dict):
        logits = self.docformer(batch_dict)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)

        loss = nn.CrossEntropyLoss()(logits, batch['label'])
        preds = torch.argmax(logits, 1)

        ## Calculating the accuracy score
        train_acc = self.train_accuracy_metric(preds, batch["label"])

        ## Logging
        self.log('train/loss', loss, prog_bar=True, on_epoch=True, logger=True, on_step=True)
        self.log('train/acc', train_acc, prog_bar=True, on_epoch=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = nn.CrossEntropyLoss()(logits, batch['label'])
        preds = torch.argmax(logits, 1)

        labels = batch['label']
        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", loss, prog_bar=True, on_step=True, logger=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True, logger=True, on_step=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        self.label.append(batch['label'])
        self.logit.append(logits)
        # return {"label": batch['label'], "logits": logits}

    def on_validation_epoch_end(self):
        # val_loss_mean = np.mean(self.training_losses)
        labels = torch.cat(self.label)
        logits = torch.cat(self.logit)
        preds = torch.argmax(logits, 1)
        wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())})
        self.logger.experiment.log(
            {"roc": wandb.plot.roc_curve(labels.cpu().numpy(), logits.cpu().numpy())})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])


## 6. Finetuning

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def main():
    datamodule = DataModule(train_ds, val_ds)
    docformer = DocFormer(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="valid/loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb.init(config=config, project="SemTabFact with DocFormer New Version")
    wandb_logger = WandbLogger(project="SemTabFact with DocFormer New Version", entity="birladivona")
    ## https://www.tutorialexample.com/implement-reproducibility-in-pytorch-lightning-pytorch-lightning-tutorial/
    pl.seed_everything(seed, workers=True)
    trainer = pl.Trainer(
        default_root_dir="logs",
        max_epochs=1,
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=True
    )
    trainer.fit(docformer, datamodule)


if __name__ == "__main__":
    main()

## References:
# 1.[DocFormer Repo](https: // github.com / uakarsh / docformer)
# 2.[MLOps Repo](https: // github.com / graviraja / MLOps-Basics)
# 3.[PyTorch Lightening Docs](https: // pytorch-lightning.readthedocs.io / en / stable / index.html)
