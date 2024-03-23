import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def create_plots(
        train_accuracy_list:list,
        train_loss_list:list,
        val_accuracy_list:list, 
        val_loss_list:list,
        task_name:str,
        epoch_list:list
          ) -> None:
    """Function to create accuracy and loss plots over epochs"""

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, train_accuracy_list, label='Train Accuracy')
    plt.plot(epoch_list, val_accuracy_list, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs Train Accuracy')
    plt.legend()
    plt.grid(linewidth=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, train_loss_list, label='Train Loss')
    plt.plot(epoch_list, val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Train Loss')
    plt.legend()
    plt.grid(linewidth=0.5)
    plt.tight_layout()

    # Save the plots
    plt.savefig(f'images/{task_name}')

    plt.show()


def create_dataloader(
        data_path:str, 
        batch_size:int=4, 
        shuffle:bool=False):
    """Function that uses the path to dataset and returns a dataloader"""
    data_path = data_path
    batch_size = batch_size

    img_width, img_height, channels = 150, 150, 3 
    resize = transforms.Resize(size=(img_height, img_width))
    dataTransforms = transforms.Compose([resize, transforms.ToTensor()])
    data = ImageFolder(data_path, transform=dataTransforms)
    BATCH_SIZE = batch_size
    dataloader = DataLoader(data,
                        batch_size=BATCH_SIZE,
                        shuffle=shuffle
                )
    return dataloader