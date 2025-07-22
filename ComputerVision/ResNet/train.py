import torch
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_seen = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item()

        useless, predicted = torch.max(outputs.data, 1)
        total_seen = total_seen + labels.size(0)
        correct_predictions = correct_predictions+ (predicted == labels).sum().item()

        progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct_predictions / total_seen
    return avg_loss, accuracy

def evaluate(model, loader, device):
    model.eval()
    correct_predictions = 0
    total_seen = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            useless, predicted = torch.max(outputs.data, 1)
            total_seen = total_seen + labels.size(0)
            correct_predictions = correct_predictions  +(predicted == labels).sum().item()
    accuracy = 100 * correct_predictions / total_seen
    return accuracy
