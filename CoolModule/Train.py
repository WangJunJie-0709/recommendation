import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm


def train(model, train_loader, val_loader, epochs, lr, log_dir):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            x = data['x']
            y = data['y']

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.view(-1, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_avg_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', train_avg_loss, epoch)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                x = data['x']
                y = data['y']

                outputs = model(x)
                loss = criterion(outputs, y.view(-1, 1))
                val_loss += loss.item()

        val_avg_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/Val', val_avg_loss, epoch)
        print(f'Epoch {epoch + 1}, Train_Loss {train_avg_loss}, Val_Loss {val_avg_loss}')

    writer.close()
