import torch
import torch.nn.functional as F
import datetime
import wandb
from evaluate import *


def train_(model, optimizer, train_dataloader, valid_dataloader, model_path,epochs=10):
    """
    training

    params:
    model (RCNN class object): rcnn model 
    optimizer:
    train_dataloader: dataloader
    valid_dataloader: dataloader
    model_path (str): path to save model
    epochs(int): number of epochs

    

    """
    if "sm" in model_path:
        model_type = "spm"
    else:
        model_type = "word tokenization"
    best_f1 = 0
    wandb.init(project="Toxic comment classification with word embedding(subword info)")
    wandb.watch(model)
    print('Start Training!')
    y_hats, targets = [], []
    for epoch in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        count = 0
        for step,batch in enumerate(train_dataloader):
            x = batch["token_ids"]
            y = torch.squeeze(batch["labels"],1)
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            y_hat = torch.max(pred, 1)[1]
            epoch_acc += (y_hat == y).sum().item()
            
            if (step+1) % 10 == 0:
                print(f'|EPOCHS| {epoch:>}/{epochs} |STEP| {step+1:>4}/{len(train_dataloader)} |LOSS| {loss.item():>.4f}')
        print(epoch_acc/len(train_dataloader.dataset) * 100)
        wandb.log({"train_loss ( "+model_type +")": epoch_loss/len(train_dataloader.dataset)})
        wandb.log({"train_acc ( "+model_type +")": epoch_acc/len(train_dataloader.dataset) * 100})
        avg_loss, accuracy, _, _, f1,_ = evaluate(model, valid_dataloader)

        wandb.log({"val_loss ( "+model_type +")": avg_loss})
        wandb.log({"val_acc ( "+model_type +")": accuracy})

        print('-'*50)
        print(f'|* VALID SET *| |VAL LOSS| {avg_loss:>.4f} |ACC| {accuracy:>.4f} |F1| {f1:>.4f}')
        print('-'*50)

        if f1 > best_f1:
            best_f1 = f1
            print(f'Saving best model... F1 score is {best_f1:>.4f}')
            torch.save(model.state_dict(), model_path)
            print('Model saved!')

