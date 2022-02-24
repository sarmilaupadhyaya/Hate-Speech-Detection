

import os
import torch
import torch.nn.functional as F





from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
def metrics(dataloader, losses, correct, y_hats, targets):
    """
    returns metrics for the result

    """
    avg_loss = losses / len(dataloader)
    accuracy = correct / len(dataloader.dataset) * 100
    precision = precision_score(targets, y_hats, average='macro')
    recall = recall_score(targets, y_hats, average='macro')
    f1 = f1_score(targets, y_hats, average='macro')
    cm = confusion_matrix(targets, y_hats)
    return avg_loss, accuracy, precision, recall, f1, cm


def evaluate(model, valid_dataloader):
    """
    evaluating only. wither for validation or test dataset
    """
    with torch.no_grad():
        model.eval()
        losses, correct = 0, 0
        y_hats, targets = [], []
        for step,batch in enumerate(valid_dataloader):
            x = batch["token_ids"]
            y = torch.squeeze(batch["labels"],1)
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            losses += loss.item()

            y_hat = torch.max(pred, 1)[1]
            y_hats += y_hat.tolist()
            targets += y.tolist()
            correct += (y_hat == y).sum().item()

    avg_loss, accuracy, precision, recall, f1, cm = metrics(valid_dataloader, losses, correct, y_hats, targets)
    return avg_loss, accuracy, precision, recall, f1, cm
