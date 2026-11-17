"""
Feel free to replace this code with your own model training code. 
This is just a simple example to get you started.

This training script uses imports relative to the project root directory.
To run this training script with uv, ensure you're in the project root directory
and execute: uv run -m submission.fashion_training
"""
import os
import torch, torchvision
import torchvision.transforms as T

from submission import engine
from submission.fashion_model import Net


def train_fashion_model(fashion_mnist, 
                        n_epochs, 
                        batch_size=32,
                        learning_rate=0.001,
                        USE_GPU=False,
                        seed=None,
                        model_kwargs=None,
                        return_state_dict_only=True,):
    """
    You can modify the contents of this function as needed, but DO NOT CHANGE the arguments,
    the function name, or the return value when return_state_dict_only=True, as this will be called during marking!
    (You can change the default values or add additional keyword arguments if needed.)
    During marking this is also run on a small dummy dataset (30 samples, 1 epoch), so it must work on that too.
    """
    # Optionally fix the random seed, so that results can be repeated
    if seed is not None:
        engine.set_seed(seed)

    # Optionally use GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create train-val split
    train_size = int(0.8 * len(fashion_mnist))
    val_size = len(fashion_mnist) - train_size
    train_split, val_split = torch.utils.data.random_split(fashion_mnist, [train_size, val_size])

    # Create datasets with transforms. You may wish to apply different transforms to the training and validation datasets
    train_data = engine.SubsetWithTransform(fashion_mnist, train_split.indices, get_transforms('train'))
    val_data = engine.SubsetWithTransform(fashion_mnist, val_split.indices, get_transforms('eval'))

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             )

    # Initialize model, loss function, and optimizer
    model = Net(**(model_kwargs or {}))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    # Added weight decay and scheduler to cut learning rate in half when accuracy plateaus
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate , weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # Training loop
    train_losses, val_losses, val_accuracies = [], [], []

    # Initializing variables to track the best accuracies and for early stopping when accuracies plateau for 7 epochs
    best_val_acc = -1.0
    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    cutoff = 7
    no_improvement = 0

    for epoch in range(n_epochs):
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")
        val_loss, accuracy = engine.evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        scheduler.step(accuracy)

        # check if accuracy improves otherwise we get closer to cutoff for early stopping
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= cutoff:
                print(f"Early stopping triggered at epoch {epoch + 1} (Best Val Acc: {best_val_acc:.4f})")
                break

    # Restore best weights into the model and move to CPU
    model.load_state_dict(best_weights)
    model.cpu()

    # Return the model's state_dict (weights) - DO NOT CHANGE THIS
    if return_state_dict_only:
        return model.state_dict()
    # Add whatever else you want to return here - return_state_dict_only=True will be enforced at runtime
    return model.state_dict(), train_losses, val_losses, val_accuracies


def get_transforms(mode='train'):
    """
    Define any data augmentations or preprocessing here if needed.
    Only standard torchvision transforms are permitted (no lambda functions, sorry), please check that 
    these pass by running model_calls.py before submission. The 'eval' transforms must be
    deterministic (no random augmentation), so only use transforms such as ToTensor, Normalize,
    Pad, Resize or CenterCrop there.
    """
    if mode == 'train':
        tfs = T.Compose([
            T.ToTensor(),
            T.RandomApply([ # Randomly apply transformations to some images
                T.RandomChoice([
                    T.RandomAffine(degrees=0, translate=(0.10, 0.10), fill=0), # Slight rotation of pixels
                    T.RandomErasing(p=1.0, scale=(0.0204, 0.0205), ratio=(1.0, 1.0), value=0.0), # Small parts of images, 4*4 are removed
                ])
            ], p=0.167),
        ])
    elif mode == 'eval': # deterministic transforms only
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), # convert images to tensors
        ])
    else:
        raise ValueError(f"Unknown mode {mode} for transforms, must be 'train' or 'eval'.")
    return tfs


def load_training_data():
    # Load FashionMNIST dataset
    # Do not change the dataset or its parameters
    print("Loading Fashion-MNIST dataset...")
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
    )
    # We load in data as the raw PIL images - recommended to have a look in visualise_dataset.ipynb! 
    return fashion_mnist


def main():
    # example usage
    # you could create a separate file that calls train_fashion_model with different parameters
    # or modify this as needed to add cross-validation, hyperparameter tuning, etc.
    fashion_mnist = load_training_data()

    # Finding the best combination of batch size and learning rate from a pre-determined list
    search_configs = [
        {'batch_size': 32, 'learning_rate': 0.01},
        {'batch_size': 32, 'learning_rate': 0.001},
        {'batch_size': 32, 'learning_rate': 0.002},
        {'batch_size': 64, 'learning_rate': 0.01},
        {'batch_size': 64, 'learning_rate': 0.001},
        {'batch_size': 64, 'learning_rate': 0.002},
    ]

    best_overall_acc = -1.0
    best_model_weights = None
    best_config = None

    for cfg in search_configs:
        print(f"\n=== Testing {cfg} ===")
        weights, _, _, val_accuracies = train_fashion_model(
            fashion_mnist,
            n_epochs=30,
            batch_size=cfg['batch_size'],
            learning_rate=cfg['learning_rate'],
            USE_GPU=True,
            seed=11,
            return_state_dict_only=False,
        )
        peak_acc = max(val_accuracies)
        best_epoch = val_accuracies.index(peak_acc) + 1
        print(f"--> Result for {cfg}: Best Val Acc = {peak_acc:.4f} (at epoch {best_epoch})")

        if peak_acc > best_overall_acc:
            best_overall_acc = peak_acc
            best_model_weights = weights
            best_config = {**cfg, 'best_epoch': best_epoch}

    print(f"\nWinning Config: {best_config} | Best Val Acc: {best_overall_acc:.4f}")

    # Save the best model weights across all searched configs
    os.makedirs('submission', exist_ok=True)
    model_save_path = os.path.join('submission', 'model_weights.pth')
    torch.save(best_model_weights, f=model_save_path)
    print(f"Saved final weights to {model_save_path}")


if __name__ == "__main__":
    main()