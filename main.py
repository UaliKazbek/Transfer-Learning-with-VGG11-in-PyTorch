import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.models as models

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights_vgg11 = models.VGG11_Weights.DEFAULT
transform = weights_vgg11.transforms()

data_set = torchvision.datasets.ImageFolder(r'C:\Users\STARLINECOMP\PycharmProjects\Pytorch\content\test_set', transform=transform)

train_set, val_set = random_split(data_set, [0.7, 0.3])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)


model = models.vgg11(weights='DEFAULT').to(device)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Linear(512*7*7, 2).to(device)
loss_model = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

EPOCH = 10
train_loss = []
train_acc = []
val_loss = []
val_acc = []
lr_list = []
best_loss = None
count = 0

for epoch in range(EPOCH):
    model.train()
    running_train_loop = []
    true_answer = 0
    train_loop = tqdm(train_loader, leave=False)
    for images, targets in train_loop:
        images = images.to(device)
        targets = targets.to(torch.long).to(device)

        pred = model(images)
        loss = loss_model(pred, targets)

        opt.zero_grad()
        loss.backward()

        opt.step()

        running_train_loop.append(loss.item())
        mean_train_loss = sum(running_train_loop)/len(running_train_loop)

        true_answer += (pred.argmax(dim=1) == targets).sum().item()

        train_loop.set_description(f'EPOCH {epoch+1}/{EPOCH}, train_loss: {mean_train_loss:.4f}')

    running_train_acc = true_answer / len(train_set)
    train_acc.append(running_train_acc)
    train_loss.append(mean_train_loss)

    model.eval()
    with torch.no_grad():
        running_val_loop = []
        true_answer = 0
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(torch.long).to(device)

            pred = model(images)
            loss = loss_model(pred, targets)

            running_val_loop.append(loss.item())
            mean_val_loss = sum(running_val_loop) / len(running_val_loop)

            true_answer += (pred.argmax(dim=1) == targets).sum().item()

        running_val_acc = true_answer / len(val_set)
        val_acc.append(running_val_acc)
        val_loss.append(mean_val_loss)

    lr_scheduler.step(mean_val_loss)
    lr = lr_scheduler._last_lr[0]
    lr_list.append(lr)

    print(f'EPOCH {epoch+1}/{EPOCH}, train_loss: {mean_train_loss:.4f}, train_acc: {running_train_acc:.4f}, val_loss: {mean_val_loss:.4f}, val_acc: {running_val_acc:.4f}, lr: {lr}')

