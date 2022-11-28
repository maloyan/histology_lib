import os

import albumentations as A
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from pkg_resources import resource_filename
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from histology_lib.dataset import TargetDataset
from histology_lib.engine import eval_fn, train_fn

accelerator = Accelerator()
device = accelerator.device

config = OmegaConf.load(resource_filename(__name__, "../configs/config.yaml"))

if accelerator.is_main_process:
    wandb.init(
        config=config,
        project=config["project"],
        name=f"{config['image_size']}_{config['model']}",
    )



transforms_train = A.Compose(
    [
        A.Resize(height=config["image_size"], width=config["image_size"], p=1),
        A.HorizontalFlip(p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.2
        ),
    ],
    p=1,
)

transforms_test = A.Compose(
    [A.Resize(height=config["image_size"], width=config["image_size"], p=1)], p=1,
)

df = pd.read_csv(config["data_csv"])
df_train, df_test  = train_test_split(df, test_size=0.2, random_state=42, stratify=df[config.target.column])
df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train[config.target.column])

train_data = []
train_target = []
for _, i in tqdm(df_train.iterrows(), total=df_train.shape[0]):
    images = [f"{i.path}/{p}" for p in os.listdir(i.path)]
    train_data.extend(
        images
    )
    train_target.extend(
        [config.target.dict[i[config.target.column]]] * len(images)
    )

valid_data = []
valid_target = []
for _, i in tqdm(df_valid.iterrows(), total=df_valid.shape[0]):
    images = [f"{i.path}/{p}" for p in os.listdir(i.path)]
    valid_data.extend(
        images
    )
    valid_target.extend(
        [config.target.dict[i[config.target.column]]] * len(images)
    )

train_dataset = TargetDataset(
    train_data,
    train_target,
    is_test=False,
    augmentation=transforms_train,
    classes=config["classes"],
)
valid_dataset = TargetDataset(
    valid_data,
    valid_target,
    is_test=False,
    augmentation=transforms_test,
    classes=config["classes"],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    drop_last=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    drop_last=False,
)

model = timm.create_model(
    config["model"], num_classes=config["classes"], pretrained=True
)
# print("PARALLEL")
# model = torch.nn.DataParallel(model, device_ids=config["device_ids"])

criterion = F.binary_cross_entropy_with_logits

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    factor=config["decay"],
    patience=config["patience"],
    verbose=True,
)

model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, valid_loader, scheduler
)

best_loss = 1000
for _ in range(config["epochs"]):
    train_loss = train_fn(train_loader, model, optimizer, criterion, accelerator)
    val_loss, metric = eval_fn(valid_loader, model, criterion)

    scheduler.step(val_loss)
    if accelerator.is_main_process:
        if val_loss < best_loss:
            print("Model saved!")
            best_loss = val_loss
            torch.save(
                model.module,
                f"{config['checkpoints']}/{config['image_size']}_{config['model']}.pt",
            )
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_metric": metric})