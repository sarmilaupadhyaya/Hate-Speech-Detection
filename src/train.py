import torch
import torch.nn.functional as F
import datetime
from evaluate import *

def train_(model, optimizer, train_dataloader, valid_dataloader, model_path,epochs=10):
    """
    training
    

    """
    best_f1 = 0
    print('Start Training!')
    for epoch in range(1, epochs+1):
        model.train()
        for step,batch in enumerate(train_dataloader):
            x = batch["token_ids"]
            y = torch.squeeze(batch["labels"],1)
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 10 == 0:
                print(f'|EPOCHS| {epoch:>}/{epochs} |STEP| {step+1:>4}/{len(train_dataloader)} |LOSS| {loss.item():>.4f}')
        print((f'|EPOCHS| {epoch:>}/{epochs} |STEP| {step+1:>4}/{len(train_dataloader)} |LOSS| {loss.item():>.4f}'))

        avg_loss, accuracy, _, _, f1, _ = evaluate(model, valid_dataloader)
        print('-'*50)
        print(f'|* VALID SET *| |VAL LOSS| {avg_loss:>.4f} |ACC| {accuracy:>.4f} |F1| {f1:>.4f}')
        print('-'*50)

        if f1 > best_f1:
            best_f1 = f1
            print(f'Saving best model... F1 score is {best_f1:>.4f}')
            torch.save(model.state_dict(), model_path)
            print('Model saved!')

