from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_device():
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def mnist_stat(train_loader):
    # We'd need to convert it into Numpy! Remember above we have converted it into tensors already
    train_data = train.train_data
    train_data = train.transform(train_data.numpy())

    print('[Train]')
    print(' - Numpy Shape:', train.train_data.cpu().numpy().shape)
    print(' - Tensor Shape:', train.train_data.size())
    print(' - min:', torch.min(train_data))
    print(' - max:', torch.max(train_data))
    print(' - mean:', torch.mean(train_data))
    print(' - std:', torch.std(train_data))
    print(' - var:', torch.var(train_data))

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    print(images.shape)
    print(labels.shape)

    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
    
    return test_loss
    

# Function to plot the training and testing graphs for loss and accuracy
def plt_fig():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    
    
def get_incorrect_predictions(model, device, test_loader):
    model.eval()
    incorrect = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            is_incorrect = pred.eq(target.view_as(pred)).item() == False

            for d, p, t, o in zip(data, pred, target, output):
                if is_incorrect:
                    incorrect.append([d.cpu(), p.cpu(), t.cpu(), o[p.item()].cpu()])
                    
    return incorrect

def plot_incorrect(incorrect, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
    total_inc = len(incorrect)
    print(f'Total Incorrect: {total_inc}')
    
    fig = plt.figure(figsize=(8, 10))
    fig.tight_layout()
    for i in range(6):
        plt.subplot(3, 3, i+1)
        plt.imshow(incorrect[i][0].numpy().squeeze(), cmap='gray_r')
        plt.title(f"Predicted: {classes[incorrect[i][1].item()]}\nActual: {classes[incorrect[i][2].item()]}")
        plt.axis('off')
        
        
        