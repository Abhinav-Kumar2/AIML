import torch
import torch.nn as nn
import torch.optim as optim
from data_handling import data_loaders
from resnet_model import ResNet, ResidualBlock
from train import train, evaluate
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    batch_size = 128
    learning_rate = 0.1

    train_loader, test_loader = data_loaders(data_dir='./data', batch_size=batch_size)

    model = ResNet(ResidualBlock, [2,2,2,2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay= 1e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75], gamma=0.1)

    batch_loss = []
    train_accuracy = []
    test_accuracy = []

    try:
        for epoch in range(100):
            loss, training_accuracy = train(model, train_loader, criterion, optimizer, device)
            testing_accuracy = evaluate(model, test_loader, device)

            batch_loss.append(loss)
            train_accuracy.append(training_accuracy)
            test_accuracy.append(testing_accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}] " f"Loss: {loss:.4f} " f"Train Acc: {training_accuracy:.3f}% " f"Test Acc: {testing_accuracy:.3f}%")

            scheduler.step()
    finally:
        epochs = list(range(1, len(batch_loss) + 1))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, batch_loss, label="Training Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss per Epoch")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracy, label="Train Accuracy", color="yellow")
        plt.plot(epochs, test_accuracy, label="Test Accuracy", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy per Epoch")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("graphs.png")
        plt.show()

if __name__ == "__main__":
    main()
