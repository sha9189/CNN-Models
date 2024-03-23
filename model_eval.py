from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import os

def get_model_score(
        model:nn.Module, 
        dataloader:DataLoader,
        criterion:any, 
        device:str="cpu"):
    """Function returns the accuracy and loss of the given model on given data"""
    model.eval()
    val_corrects = 0
    val_total = 0
    val_losses = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader), 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            preds = ((outputs > 0.5) * 1.0)
            loss = criterion(outputs.float(), labels.float())

            val_corrects += torch.sum(preds == labels.data)
            val_total += len(preds)
            val_losses.append(loss)
    val_accuracy = (val_corrects / val_total).item()
    val_loss = (sum(val_losses) / len(val_losses)).item()
    return val_accuracy, val_loss


def get_latest_checkpoint_number(task_name:str) -> int:
    checkpoint_dir = f'checkpoints/{task_name}'
    latest_checkpoint_number = max(
        [int(f[11:-3]) for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch')],
        default=None
    )
    return latest_checkpoint_number


def get_latest_checkpoint(task_name:str) -> any :
    """Returns the latest checkpoint for a task"""
    checkpoint_dir = f'checkpoints/{task_name}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return None
    latest_checkpoint_number = max(
        [int(f[11:-3]) for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch')],
        default=None
    )
    if latest_checkpoint_number:
        latest_checkpoint = get_checkpoint_by_model_number(
            task_name=task_name,
            model_number=latest_checkpoint_number)
        return latest_checkpoint
    return None

def get_checkpoint_by_model_number(
        task_name:str, 
        model_number:int):
    checkpoint_dir = f'checkpoints/{task_name}'
    checkpoint_name = f"model_epoch{model_number}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(f"Checkpoint {model_number} loaded")
        return checkpoint 
    return None

