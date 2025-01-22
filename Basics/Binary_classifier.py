import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import matplotlib.pyplot as plt


class BinaryDataset(data.Dataset):
    def __init__(self, size, std=0.05, radius=1.0, separation=1.5):
        """
        Inputs:
        - size : Number of samples
        - std : Standard deviation of the noise
        """
        super().__init__()
        self.size = size
        self.std = std
        self.radius = radius
        self.separation = separation
        self.generate_data()

    def generate_data(self):
        """
        Generates 2D circular data points for binary classification.
        Class 0 points are closer to the origin, and Class 1 points are farther.
        """
        n_class0 = self.size // 2
        n_class1 = self.size - n_class0

        # Class 0: Inner circle
        angles_class0 = 2 * torch.pi * torch.rand(n_class0)
        radii_class0 = self.radius * torch.rand(n_class0)
        x_class0 = radii_class0 * torch.cos(angles_class0)
        y_class0 = radii_class0 * torch.sin(angles_class0)

        # Class 1: Outer circle
        angles_class1 = 2 * torch.pi * torch.rand(n_class1)
        radii_class1 = self.radius * self.separation + self.radius * torch.rand(n_class1)
        x_class1 = radii_class1 * torch.cos(angles_class1)
        y_class1 = radii_class1 * torch.sin(angles_class1)

        # Combine data
        data_class0 = torch.stack([x_class0, y_class0], dim=1)
        data_class1 = torch.stack([x_class1, y_class1], dim=1)
        labels_class0 = torch.zeros(n_class0, dtype=torch.long)
        labels_class1 = torch.ones(n_class1, dtype=torch.long)

        # Add noise
        data_class0 += self.std * torch.randn_like(data_class0)
        data_class1 += self.std * torch.randn_like(data_class1)

        # Concatenate all data
        self.data = torch.cat([data_class0, data_class1], dim=0)
        self.label = torch.cat([labels_class0, labels_class1], dim=0)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label


class BinaryClassifier(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act = nn.Sigmoid()
        self.linear2 = nn.Linear(num_hidden, num_outputs)
        self.output_act = nn.Sigmoid()  # Add Sigmoid to output for probability

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.output_act(x)  # Apply sigmoid to output
        return x


def train(model, optimizer, data_loader, num_epochs):
    model.train()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    loss_module = nn.BCELoss()
    tqdm_bar = tqdm(total=num_epochs, desc="Training Progress", position=0)  # Single tqdm bar

    for epoch in range(num_epochs):
        epoch_loss = 0  # For tracking loss per epoch
        for inputs, labels in data_loader:
            # Data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Run model
            preds = model(inputs)
            preds = preds.squeeze(dim=1)  # Adjust dimensions for BCELoss

            # Compute loss
            loss = loss_module(preds, labels.float())
            epoch_loss += loss.item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Update
            optimizer.step()

        # Update tqdm bar
        tqdm_bar.set_postfix({'Epoch Loss': f"{epoch_loss / len(data_loader):.4f}"})
        tqdm_bar.update(1)

    tqdm_bar.close()


def visualize_results(model, dataset):
    """
    Visualizes the classification results of the trained model.
    """
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    data = dataset.data.to(device)
    labels = dataset.label.numpy()

    with torch.no_grad():
        preds = model(data).squeeze(dim=1).cpu().numpy()
        preds = (preds >= 0.5).astype(int)  # Threshold at 0.5

    # Visualization
    plt.figure(figsize=(6, 6))
    plt.scatter(data.cpu().numpy()[labels == 0][:, 0], data.cpu().numpy()[labels == 0][:, 1], color='red', label='Class 0 (True)')
    plt.scatter(data.cpu().numpy()[labels == 1][:, 0], data.cpu().numpy()[labels == 1][:, 1], color='blue', label='Class 1 (True)')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.title("Classification Results")
    plt.show()


if __name__ == '__main__':
    def main():
        # Model, dataset, and data loader
        model = BinaryClassifier(num_inputs=2, num_hidden=10, num_outputs=1)
        dataset = BinaryDataset(size=1000, std=0.1)
        data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Train the model
        train(model, optimizer, data_loader, num_epochs=200)

        # Visualize results
        visualize_results(model, dataset)

    main()
