## Importing the libraries
#System libraries
import wandb
import warnings
import os
import sys
from pathlib import Path

#Dataframes and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Sklearn metrics
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix

#Pytorch and huggingface
import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

## Adding the path of docformer to system path
sys.path.append('./docformer/src/docformer/')
# sys.path.append('/gpfs/home1/ibirlad/.conda/envs/semtabfact_venv/bin/tesseract')

## Importing the functions from the DocFormer Repo
from dataset import create_features
from modeling import DocFormerEncoder, ResNetFeatureExtractor, DocFormerEmbeddings, LanguageFeatureExtractor, DocFormer
from transformers import BertTokenizerFast


## Logging into wandb

wandb.login(key="X") #Add your wandb key

#Environment setup
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


## Global variables
seed = 42
target_size = (500, 384)
n_classes = 3

## Setting some hyperparameters
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(torch.cuda.is_available())
# print(f"TORCH VERSION: {torch.version.cuda}")

## One can change this configuration and try out new combination
config = {
    "coordinate_size": 96,  ## (768/8), 8 for each of the 8 coordinates of x, y
    "hidden_dropout_prob": 0.2,
    "hidden_size": 768,
    "image_feature_pool_shape": [7, 7, 256],
    "intermediate_ff_size_factor": 4,
    "max_2d_position_embeddings": 1000,
    "max_position_embeddings": 512,
    "max_relative_positions": 8,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "shape_size": 96,
    "vocab_size": 30522,
    "layer_norm_eps": 1e-12,
    "classes": 3
}


## Importing the data
ROOT_DIRECTORY_PATH = str(Path(__file__).parent)
PNG_PATH_TRAIN = "png_data/data_10000.csv"
PNG_PATH_TEST = "png_data_test/data_test.csv"

###### Just manually annotated data ##########################################
# PNG_PATH = "png_data/data_aug.csv"
# data = pd.read_csv(os.path.join(ROOT_DIRECTORY_PATH, PNG_PATH), index_col=0)
# data["table_name"] = ROOT_DIRECTORY_PATH + "/" + data["table_name"]
##############################################################################

#combine data from manual and automatic annotations
data_man = pd.read_csv(os.path.join(ROOT_DIRECTORY_PATH, PNG_PATH_TRAIN), index_col=0)
data_test = pd.read_csv(os.path.join(ROOT_DIRECTORY_PATH, PNG_PATH_TEST), index_col=0)

data["table_name"] = ROOT_DIRECTORY_PATH + "/" + data["table_name"]
data_test["table_name"] = ROOT_DIRECTORY_PATH + "/" + data_test["table_name"]
train_df = data.reset_index(drop=True)
valid_df = data_test.reset_index(drop=True)



## 3. Making the dataset

## Creating the dataset

class SemTabFactData(Dataset):

    def __init__(self, image_list, table_list, label_list, statement_list, target_size, tokenizer, max_len=512, transform=None):

        self.image_list = image_list
        self.table_list = table_list
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
        table_path = self.table_list[idx]
        label = self.label_list[idx]
        statement = self.statement_list[idx]

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
            use_ocr=True
        )
        if self.transform is not None:
            final_encoding['resized_scaled_img'] = self.transform(final_encoding['resized_scaled_img'])

        keys_to_reshape = ['x_features', 'y_features', 'resized_and_aligned_bounding_boxes']
        for key in keys_to_reshape:
            final_encoding[key] = final_encoding[key][:self.max_len]

        final_encoding['label'] = torch.as_tensor(label).long()

        table_encoding = tokenizer(table_path,
                                   padding="max_length",
                                   max_length=self.max_len,
                                   truncation=True)

        final_encoding['input_ids'] = torch.as_tensor(table_encoding["input_ids"])

        statement_encoding = tokenizer(statement,
                                       padding="max_length",
                                       max_length=self.max_len,
                                       truncation=True)

        final_encoding['statement'] = torch.as_tensor(statement_encoding["input_ids"])
        return final_encoding


## Defining the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

## Normalization to mean and std
transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

train_ds = SemTabFactData(train_df['table_name'].tolist(), train_df['csv_name'].tolist(), train_df['label'].tolist(), train_df['statement'].tolist(),
                          target_size, tokenizer, config['max_position_embeddings'], transform)
val_ds = SemTabFactData(valid_df['table_name'].tolist(), valid_df['csv_name'].tolist(), valid_df['label'].tolist(), valid_df['statement'].tolist(),
                        target_size, tokenizer, config['max_position_embeddings'], transform)

### Collate Function:
# from [here](https: // stackoverflow.com / questions / 65279115 / how - to - use - collate - fn - with-dataloaders)

def collate_fn(data_bunch):
    '''
    A function for the dataloader to return a batch dict of given keys

    data_bunch: List of dictionaries
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

class DataModule(pl.LightningDataModule):

    def __init__(self, train_dataset, val_dataset, batch_size=16):
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

## Loading from checkpoint

class DocFormerForMLM(DocFormer):

    def __init__(self, config):
        super().__init__(config)
        self.mlm_token = nn.Linear(config['hidden_size'], config['vocab_size'])

    def forward(self, batch):
        output = super().forward(batch)
        mlm_token = self.mlm_token(output)
        return {"mlm_token": mlm_token}


class PLModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.vocab_size = config['vocab_size']
        self.model = DocFormerForMLM(config)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss_fn(batch["labels"], output["mlm_token"])['mlm_loss']

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)

        loss = self.loss_fn(batch["labels"], output["mlm_token"])['mlm_loss']

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)


ckpt_path = "./docformer_best_ckpt.ckpt"
model = PLModel.load_from_checkpoint(ckpt_path, map_location="cpu", config=config)

# Modeling part


## 5. Modeling Part

class DocFormerForClassification(nn.Module):

    def __init__(self, config):
        super(DocFormerForClassification, self).__init__()

        self.resnet = ResNetFeatureExtractor(hidden_dim=config['max_position_embeddings'])
        self.resnet.load_state_dict(model.model.extract_feature.visual_feature.state_dict())

        self.embeddings = DocFormerEmbeddings(config)
        self.embeddings.load_state_dict(model.model.extract_feature.spatial_feature.state_dict())

        self.lang_emb = LanguageFeatureExtractor()
        self.lang_emb.load_state_dict(model.model.extract_feature.language_feature.state_dict())

        self.config = config
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.linear_layer = nn.Linear(in_features=config['hidden_size'], out_features=n_classes)  ## Number of Classes
        self.encoder = DocFormerEncoder(config)
        self.linear = nn.Linear(2*config['hidden_size'], config['hidden_size'])


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
        text = torch.cat([table_emb, statement_emb], dim=-1)
        t_bar = self.linear(text)
        out = self.encoder(t_bar, v_bar, t_bar_s, v_bar_s)
        out = self.linear_layer(out)
        out = out[:, 0, :]
        return out


## Defining pytorch lightning model


class DocFormer(pl.LightningModule):

    def __init__(self, config, model_name = "docformer_base_noOCR", lr=1e-3):
        super(DocFormer, self).__init__()

        self.save_hyperparameters()
        self.config = config
        self.docformer = DocFormerForClassification(config)
        self.lr = lr

        self.num_classes = n_classes
        self.train_accuracy_metric = torchmetrics.Accuracy(task='multiclass',
                                                           num_classes=self.num_classes)
        self.val_accuracy_metric = torchmetrics.Accuracy(task='multiclass',
                                                         num_classes=self.num_classes)
        self.f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes)

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
        self.log('train/loss', loss, prog_bar=True, on_epoch=True, logger=True)
        self.log('train/acc', train_acc, prog_bar=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = nn.CrossEntropyLoss()(logits, batch['label'])
        preds = torch.argmax(logits, 1)

        labels = batch['label']
        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", loss, prog_bar=True, on_step=True, logger=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True, logger=True, on_step=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        self.label.append(batch['label'])
        self.logit.append(logits)
        return loss

    def on_validation_epoch_end(self):
        labels = torch.cat(self.label)
        logits = torch.cat(self.logit)
        preds = torch.argmax(logits, 1)
        self.logger.experiment.log(
            {"roc": wandb.plot.roc_curve(labels.cpu().numpy(), logits.cpu().numpy())})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])


## 6. Finetuning

def main():
    datamodule = DataModule(train_ds, val_ds)
    docformer = DocFormer(config)
    MODELPATH = "./models"


    checkpoint_callback = ModelCheckpoint(
        monitor="valid/loss", dirpath=MODELPATH, mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=5, verbose=True, mode="min"
    )

    wandb.init(config=config, project="SemTabFact with DocFormer Snel")
    wandb_logger = WandbLogger(project="SemTabFact with DocFormer Snel", entity="birladivona")
    pl.seed_everything(seed, workers=True)
    trainer = pl.Trainer(
        default_root_dir="logs",
        max_epochs=20,
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=True,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        num_nodes=1
    )
    if torch.cuda.is_available():
        docformer = docformer.to(device)
    trainer.fit(docformer, datamodule)
    return docformer


if __name__ == "__main__":
    docformer = main()



## References:
# 1.DocFormer Repo: https: // github.com / uakarsh / docformer
# 2.MLOps Repo: https: // github.com / graviraja / MLOps-Basics
# 3.PyTorch Lightening Docs: https: // pytorch-lightning.readthedocs.io / en / stable / index.html
