import os
import argparse
import torch
from model import ConvLSTM_EncDec
from torch import nn, optim
from torch.utils.data import DataLoader
from dataloader import TyphoonDataset, save_normalization_params
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import numpy as np


def get_dataloaders(track_root, precip_root, meteo_root,
                    T_in=12, T_out=12, batch_size=8):

    norm_param_file = 'norm_params.json'

    full_dataset = TyphoonDataset(
        track_root=track_root,
        precip_root=precip_root,
        meteo_root=meteo_root,
        years=list(range(1998, 2024)),
        T_in=T_in,
        T_out=T_out,

        normalize_input=False,
        normalize_target=False,

        norm_param_file=None,

        add_noise=False
    )

    full_samples = full_dataset.samples

    total_indices = list(range(len(full_samples)))

    train_indices, temp_indices = train_test_split(
        total_indices,
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=2/3,
        random_state=42,
        shuffle=True
    )

    train_dataset = TyphoonDataset(
        track_root=track_root,
        precip_root=precip_root,
        meteo_root=meteo_root,

        samples=full_samples,

        selected_indices=train_indices,

        T_in=T_in,
        T_out=T_out,

        normalize_input=True,
        normalize_target=True,

        norm_param_file=None,

        add_noise=False
    )

    save_normalization_params(
        norm_param_file,
        train_dataset.input_mean,
        train_dataset.input_std,
        train_dataset.cmorph_min,
        train_dataset.cmorph_max
    )

    val_dataset = TyphoonDataset(
        track_root=track_root,
        precip_root=precip_root,
        meteo_root=meteo_root,

        samples=full_samples,

        selected_indices=val_indices,

        T_in=T_in,
        T_out=T_out,

        normalize_input=True,
        normalize_target=True,

        norm_param_file=norm_param_file,

        add_noise=False
    )

    test_dataset = TyphoonDataset(
        track_root=track_root,
        precip_root=precip_root,
        meteo_root=meteo_root,

        samples=full_samples,

        selected_indices=test_indices,

        T_in=T_in,
        T_out=T_out,

        normalize_input=True,
        normalize_target=True,

        norm_param_file=norm_param_file,

        add_noise=False
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


def evaluate_model(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

def train_model(model, train_loader, eval_loader, criterion, optimizer, device, num_epochs=150):
    best_loss = float('inf')

    with open('loss.txt', 'w') as f:
        f.write('epoch,train_loss,eval_loss,test_loss\n')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            progress_bar.set_postfix(loss=loss.item())

        average_loss = total_loss / total_samples
        print(f'Epoch {epoch + 1}, Train Loss: {average_loss:.7f}')

        eval_loss = evaluate_model(model, eval_loader, criterion, device)
        print(f'Epoch {epoch + 1}, Eval Loss: {eval_loss:.7f}')

        with open('loss.txt', 'a') as f:
            f.write(f'{epoch + 1},{average_loss:.7f},{eval_loss:.7f}')

        if eval_loss < best_loss:
            best_loss = eval_loss
            print(f'New best model found at epoch {epoch + 1} with loss {best_loss:.7f}. Saving model...')
            torch.save(model.state_dict(), 'best_model_weights_tc.pth')

    print("Training complete.")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(42)

    train_loader, eval_loader, test_loader = get_dataloaders(
        track_root=args.track_root,
        precip_root=args.precip_root,
        meteo_root=args.meteo_root,
        T_in=args.T_in,
        T_out=args.T_out,
        batch_size=args.batch_size,
    )

    model = ConvLSTM_EncDec(shape_in=(12, 6, 80, 80))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_model(model, train_loader, eval_loader, test_loader, criterion, optimizer, device, num_epochs=args.num_epochs)

    model.load_state_dict(torch.load('best_model_weights_tc.pth'))

    test_loss = test_model(model, test_loader, criterion, device)
    print(f'Final Test Loss: {test_loss:.7f}')

    with open('loss.txt', 'a') as f:
        f.write(f'final_test_loss,{test_loss:.7f}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PISTN model on TC_precip data.")
    parser.add_argument('--track_root', type=str, default='')
    parser.add_argument('--precip_root', type=str, default='')
    parser.add_argument('--meteo_root', type=str, default='')
    parser.add_argument('--T_in', type=int, default=12, help='Number of input time steps.')
    parser.add_argument('--T_out', type=int, default=12, help='Number of output time steps.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--reshape', type=int, nargs='+', help='Optional reshape permutation.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs.')


    args = parser.parse_args()
    main(args)
