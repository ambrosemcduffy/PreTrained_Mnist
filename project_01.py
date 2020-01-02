import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision import models, datasets, transforms

train_dir = 'flower_photos/train/'
test_dir = 'flower_photos/test/'
batch_size = 20
transform = transforms.Compose([transforms.RandomResizedCrop(244),
                                transforms.ToTensor()])

# Importing in the data
train_data = datasets.ImageFolder(train_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True)

# Assigning the clases names to a list
classes = ['daisy', 'dandelion', 'roses', 'sunflowr', 'tulips']

# Exploring the data
data, target = next(iter(train_loader))
data = data.numpy()
fig = plt.figure(figsize=(25, 4))

for i in np.arange(20):
    ax = fig.add_subplot(2,
                         20/2,
                         i+1,
                         xticks=[],
                         yticks=[])
    plt.imshow(np.transpose(data[i], (1, 2, 0)))
    ax.set_title(classes[target[i]])

# Importing the pretrained network
vgg16 = models.vgg16(pretrained=True)

# Freezing the weights
for param in vgg16.features.parameters():
    param.requires_grad = False

# Configuring the final layer class length
n_inputs = vgg16.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes))
vgg16.classifier[6] = last_layer

# setting an optimizer & criterion
optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# model will train on gpu if there is one.
if torch.cuda.is_available():
    vgg16.cuda()

epochs = 3
print_every = 19
# Training the model

for epoch in range(epochs):
    train_loss = 0.0
    for batch_i, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = vgg16.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_loss = train_loss/train_loader.batch_size
        if batch_i % 20 == 19:
            print("epoch {}, batch {}, trainloss {}".format(epoch+1,
                  batch_i+1,
                  train_loss))
            train_loss = 0.0

torch.save(vgg16.state_dict(), 'saved_model/checkpoint.pth')


def overall_accuracy():
    test_loss = 0.0
    class_correct = list(0. for i in range(20))
    class_total = list(0. for i in range(20))
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        output = vgg16(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, dim=1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(20):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    overall = np.sum(class_correct)/np.sum(class_total)
    test_loss = test_loss/len(test_loader.dataset)
    return test_loss, overall


test_loss, acc = overall_accuracy()

print("Overall accuracy is {}%".format(str(acc*100)[:2]))
