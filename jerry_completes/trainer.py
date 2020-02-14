from tqdm import tqdm
import torch


def train_epoch(model, train_dataloader, optimizer, scheduler, device):
    model.train()

    total_loss = 0.0
    steps = 0
    for batch in tqdm(train_dataloader, desc="Iteration"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs, labels=labels)
        loss = outputs[0]
        loss.backward()

        total_loss += loss.mean().item()
        steps += 1
        optimizer.step()
        scheduler.step()

    return total_loss / steps, scheduler.get_last_lr()[0]


def valid_epoch(model, valid_dataloader, device):
    model.eval()

    total_loss = 0.0
    steps = 0
    for batch in tqdm(valid_dataloader, desc="Iteration"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            total_loss += lm_loss.mean().item()
        steps += 1

        total_loss = total_loss / steps

    return total_loss
