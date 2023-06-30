import wandb
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os

#Sklearn metrics
from sklearn.utils import shuffle

#Pytorch and huggingface
import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

## Adding the path of docformer to system path
sys.path.append('./docformer/src/docformer/')
# sys.path.append('/gpfs/home1/ibirlad/.conda/envs/semtabfact_venv/bin/tesseract')

## Importing the functions from the DocFormer Repo
from dataset import create_features
from modeling import DocFormerEncoder, ResNetFeatureExtractor, DocFormerEmbeddings, LanguageFeatureExtractor, DocFormer
from transformers import BertTokenizerFast
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import evaluate


#Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## Global variables
seed = 42
target_size = (500, 384)
n_classes = 3

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

class DocFormer(pl.LightningModule):

    def __init__(self, config, model_name = "docformer_base", lr=1e-3):
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
        # val_loss_mean = np.mean(self.training_losses)
        labels = torch.cat(self.label)
        logits = torch.cat(self.logit)
        preds = torch.argmax(logits, 1)
        self.logger.experiment.log(
            {"roc": wandb.plot.roc_curve(labels.cpu().numpy(), logits.cpu().numpy())})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])



config = {
    "coordinate_size": 96,
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

docformer = DocFormer(config)

## Importing the data
ROOT_DIRECTORY_PATH = os.path.abspath("")
PNG_PATH_MAN = "png_data/data_aug.csv"
PNG_PATH_AUTO = "png_data_auto/data_aug_auto.csv"
PNG_PATH_TEST = "png_data_test/data_test.csv"

#combine data from manual and automatic annotations
data_man = pd.read_csv(os.path.join(ROOT_DIRECTORY_PATH, PNG_PATH_MAN), index_col=0)
data_auto = pd.read_csv(os.path.join(ROOT_DIRECTORY_PATH, PNG_PATH_AUTO), index_col=0)
data_test = pd.read_csv(os.path.join(ROOT_DIRECTORY_PATH, PNG_PATH_TEST), index_col=0)

data_auto = shuffle(data_auto, random_state=42, n_samples=3872) #Only use 10000 samples in total for training time optimization
data = pd.concat([data_man, data_auto])
data = shuffle(data, random_state=42)
data.reset_index(inplace=True, drop=True)

data["table_name"] = ROOT_DIRECTORY_PATH + "/" + data["table_name"]
data_test["table_name"] = ROOT_DIRECTORY_PATH + "/" + data_test["table_name"]

train_df = data.reset_index(drop=True)
valid_df = data_test.reset_index(drop=True)

CHECKPOINT_PATH = "models/epoch=16-step=2006.ckpt"
model_path = os.path.join(ROOT_DIRECTORY_PATH, CHECKPOINT_PATH)
pl_model = docformer.load_from_checkpoint(model_path)


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

## Normalization to mean and std
transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

train_ds = SemTabFactData(train_df['table_name'].tolist(), train_df['label'].tolist(), train_df['statement'].tolist(),
                          target_size, tokenizer, config['max_position_embeddings'], transform)
val_ds = SemTabFactData(valid_df['table_name'].tolist(), valid_df['label'].tolist(), valid_df['statement'].tolist(),
                        target_size, tokenizer, config['max_position_embeddings'], transform)


def collate_fn(data_bunch):

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

#Load model and evaluate
true_labels = []
true_predictions = []

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_metric = evaluate.load("precision", average= "micro")
pl_model.eval()

model = pl_model.to(device)

for idx, batch in enumerate(tqdm(datamodule.val_dataloader())):
    # move batch to device
    batch = {k:v.to(device) for k,v in batch.items()}
    with torch.no_grad():
      outputs = model.forward(batch)

    predictions = outputs.argmax(-1)
    true_predictions.append(predictions)
    true_labels.append(batch["label"])
    eval_metric.add_batch(references=predictions, predictions=batch["label"])
    eval_metric.compute(average= "micro")


for key in ['precision', 'recall', 'f1']:
    eval_metric = evaluate.load(key, average = "micro")
    for i in range(len(true_labels)):
        eval_metric.add_batch(references=true_labels[i], predictions=true_predictions[i])
    print(eval_metric.compute(average = "micro"))


#For binary model
# disp = ConfusionMatrixDisplay(confusion_matrix=true_predictions,
#                               display_labels=np.asarray(["R","E"]))
#
# disp.plot(cmap=plt.cm.Greys)
# plt.savefig('cm2.pdf')

#For 3-class model
disp = ConfusionMatrixDisplay(confusion_matrix=true_predictions,
                              display_labels=np.asarray(["R","E","U"]))

disp.plot(cmap=plt.cm.Greys)
plt.savefig('cm3.pdf')
