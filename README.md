# Transfer Learning with VGG11 in PyTorch
This script implements transfer learning using VGG11 pretrained on ImageNet, fine-tuned on a custom dataset (e.g., cats vs dogs).

## Features
- Uses torchvision.models.vgg11 with pretrained weights
- Freezes all layers except the final classifier
- Replaces classifier with nn.Linear(512 * 7 * 7, 2)
- Includes:
- Custom dataset loading
- Training and validation loop
- Learning rate scheduler ReduceLROnPlateau
- Accuracy tracking per epoch

## Training
Key parts of the training:
``` text
# Load pretrained model
model = models.vgg11(weights='DEFAULT')

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier = nn.Linear(512 * 7 * 7, 2)
```
##  Output Example
EPOCH 10/10, train_loss: 0.0001, train_acc: 1.0000, val_loss: 0.1391, val_acc: 0.9818, lr: 0.0001
