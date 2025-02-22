from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import os
import pickle 
from models.catsvsdogsmodel import CatsVsDogsModel
from helpers import create_plots, create_dataloader
from model_eval import get_latest_checkpoint_number, get_latest_checkpoint

task_name = "task1"
EPOCHS = 20
BATCH_SIZE = 4
lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device}")

# load data for training
train_data_path = "data/train/"
val_data_path = "data/val/"
test_data_path = "data/test/"

train_dataloader = create_dataloader(
    data_path=train_data_path,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataloader = create_dataloader(
    data_path=val_data_path,
    batch_size=BATCH_SIZE
)

test_dataloader = create_dataloader(
    data_path=test_data_path,
    batch_size=BATCH_SIZE
)

# Let's check out what we've created
print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of val dataloader: {len(val_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")


# load model
model = CatsVsDogsModel()
model = model.to(device)

# load loss function
criterion = nn.BCELoss()

# load optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

# lists to capture performance metrics
epoch_losses = []
epoch_accuracys = []
epoch_val_accuracys = []
epoch_val_losses = []

checkpoint_dir = f'checkpoints/{task_name}'

metrics_file_path = lambda metric: f"{checkpoint_dir}/{task_name}-{metric}.pkl"

losses_file_path = metrics_file_path("train-losses")
if os.path.exists(losses_file_path):
    with open(losses_file_path, 'rb') as f:
        epoch_losses = pickle.load(f)
    with open(metrics_file_path("train-accuracys"), 'rb') as f:
        epoch_accuracys = pickle.load(f)
    with open(metrics_file_path("val-accuracys"), 'rb') as f:
        epoch_val_accuracys = pickle.load(f)
    with open(metrics_file_path("val-losses"), 'rb') as f:
        epoch_val_losses = pickle.load(f)
    print("Performance metrics for last epoch loaded")
else:
    print("No file found for past performance metrics")


# load latest checkpoint if it exists
latest_checkpoint = get_latest_checkpoint(task_name=task_name)

last_epoch = 0
if latest_checkpoint:
    model.load_state_dict(latest_checkpoint['model_state_dict'])
    optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])
    last_epoch = latest_checkpoint['epoch']
    loss = latest_checkpoint['loss']

# training
for epoch in range(last_epoch + 1, EPOCHS + 1):
    preds_list = []
    labels_list = []
    losses = []
    for i, data in enumerate(tqdm(train_dataloader), 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()
        preds = ((outputs > 0.5) * 1.0)

        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()

        preds_list += list(preds)
        labels_list += list(labels.data)
        losses.append(loss.item())
    epoch_corrects = sum([1 for a, b in zip(preds_list, labels_list) if a==b])
    accuracy = epoch_corrects / len(labels_list)
    epoch_accuracys.append(accuracy)
    epoch_loss = sum(losses) / len(losses)
    epoch_losses.append(epoch_loss)

    # model evaluation
    model.eval()
    val_corrects = 0
    val_total = 0
    val_losses = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader), 0):
            inputs, labels = data
            # ACTIVATE ON GPU
            # inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs).squeeze()
            preds = ((outputs > 0.5) * 1.0)
            loss = criterion(outputs.float(), labels.float())
            
            val_corrects += torch.sum(preds == labels.data) 
            val_total += len(preds)
            val_losses.append(loss)
    epoch_val_accuracy = (val_corrects / val_total).item()
    epoch_val_accuracys.append(epoch_val_accuracy)
    epoch_val_loss = (sum(val_losses) / len(val_losses)).item()
    epoch_val_losses.append(epoch_val_loss)
    print(epoch_accuracys, epoch_losses, epoch_val_accuracys, epoch_val_losses)

    # Save model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Model saved after {epoch} epochs.")
    
    # save performance metrics
    with open(metrics_file_path("train-losses"), 'wb') as f:
        pickle.dump(epoch_losses, f)
    with open(metrics_file_path("train-accuracys"), 'wb') as f:
        pickle.dump(epoch_accuracys, f)
    with open(metrics_file_path("val-accuracys"), 'wb') as f:
        pickle.dump(epoch_val_accuracys, f)
    with open(metrics_file_path("val-losses"), 'wb') as f:
        pickle.dump(epoch_val_losses, f)
    print("Performace metrics saved")

# create training plots
create_plots(
    train_accuracy_list=epoch_accuracys,
    train_loss_list=epoch_losses, 
    val_accuracy_list=epoch_val_accuracys,
    val_loss_list=epoch_val_losses,
    task_name=f"{task_name}",    
    epoch_list=list(range(1, len(epoch_accuracys)+1))
)


# epoch - learning rate
# 1 - 0.001
# 2 - 0.001
# 3 - 0.001
# 4 - 0.001
# 5 - 0.001
# 6 - 0.001
# 7 - 0.001
# 8 - 0.001
# 9 - 0.001
# 10 - 0.001
# 11 - 0.0001
# 12 - 0.0001
# 13 - 0.0001
# 14 - 0.0001
# Switch to adam optimizer
# 15 - 0.001
# 16 - 0.001
# 17 - 0.001
# 18 - 0.001
# 19 - 0.001
# 20 - 0.001
