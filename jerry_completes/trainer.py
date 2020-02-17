import torch
from tqdm import tqdm


def train_epoch(model, train_dataloader, optimizer, scheduler, device):
    model.zero_grad()
    model.train()

    total_loss = 0.0
    steps = 0
    train_iterator = tqdm(train_dataloader, desc="Training")
    for batch in train_iterator:
        inputs, labels = (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs, labels=labels)
        loss = outputs[0]
        loss.backward()

        total_loss += loss.item()
        steps += 1

        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()
        model.zero_grad()
    train_iterator.close()

    return total_loss / steps, scheduler.get_last_lr()[0]


def valid_epoch(model, valid_dataloader, device):
    model.eval()

    total_loss = 0.0
    steps = 0
    valid_iterator = tqdm(valid_dataloader, desc="Validating")
    for batch in valid_iterator:
        inputs, labels = (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            total_loss += lm_loss.item()
        steps += 1
    valid_iterator.close()

    return total_loss / steps
