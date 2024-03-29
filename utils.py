from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def GetIncorrectPreds(data, pPrediction, pLabels):
  images = []
  incorrectPreds = []
  nonMatchingLabels = []
  # print("pPrediction type:", type(pPrediction), "Shape:", pPrediction.shape)
  # print("pLabels type:", type(pLabels), "Shape:", pLabels.shape)
  preds = pPrediction.argmax(dim=1)
  indexes = pLabels.ne(pPrediction.argmax(dim=1))
  for image, pred, label in zip(data, preds, pLabels):
      if pred.ne(label):
          images.append(image.cpu())
          incorrectPreds.append(pred.cpu().item())
          nonMatchingLabels.append(label.cpu().item())

  # print("Incorrect Preds:", incorrectPreds, "Labels:", nonMatchingLabels)
  return images, incorrectPreds, nonMatchingLabels

def incorrectOutcomes(model, device, test_loader,reqData):
    model.eval()

    test_loss = 0
    correct = 0
    incorrectPreds = []
    nonMatchingLabels = []
    images = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            imageSet, incPred, nonMatchLabel = GetIncorrectPreds(data, output, target)
            nonMatchingLabels = nonMatchingLabels + nonMatchLabel
            incorrectPreds = incorrectPreds + incPred
            images = images + imageSet

            if len(incorrectPreds) > reqData:
              break

    return images, nonMatchingLabels, incorrectPreds

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc= 100*correct/processed
  print('\nProcessed: {}, Len TrainLoader: {}'.format(processed, len(train_loader)))
  # train_losses.append(train_loss/len(train_loader))
  train_loss = train_loss/len(train_loader)
  print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
      train_loss, correct, len(train_loader.dataset),
      100. * correct / len(train_loader.dataset)))

  
  return train_acc, train_loss

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc = (100. * correct / len(test_loader.dataset))
    # test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_acc, test_loss

def drawLossAccuracyPlots(train_losses, train_accs, test_losses, test_accs):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accs)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accs)
    axs[1, 1].set_title("Test Accuracy")
