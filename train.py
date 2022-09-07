# Train 함수 정의
import torch


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def training(model, dataloader, criterion,  optimizer, scheduler):

    model.to(device)
    model.train()

    train_accuracy = 0.0
    train_loss = 0.0
    data_len = 0

      # Training
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Making the gradients 0 at the start of a new batch

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()

        train_loss += loss.data * images.size(0)  
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))
        data_len += len(prediction)

        # Print loss, accuracy of every batch*25th batch
        if i % (25*len(prediction)) == 0:
            batch_loss = train_loss / data_len
            batch_accuracy = train_accuracy / data_len

            print(f'batch {i // len(prediction)} : train loss {batch_loss} , '
                  f'train accuracy {100*batch_accuracy:.4f}%')

    # Sum of len(mini batch) == data_len
    train_loss = train_loss / data_len
    train_accuracy = train_accuracy / data_len
    scheduler.step(loss)  # Update the weight and bias

    return train_loss, train_accuracy