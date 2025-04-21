# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Construct a binary classification model leveraging a pretrained VGG19 architecture to differentiate between defected and non-defected capacitors by modifying the final layer to a single neuron. Train the model using a dataset comprising images of capacitors with and without defects to enhance detection accuracy. Optimize and assess the model to ensure robust performance in capacitor quality assessment for manufacturing applications.

## DESIGN STEPS
# STEP 1:
Gather and preprocess a dataset containing images of defected and non-defected capacitors, ensuring proper data augmentation and normalization.
# STEP 2:
Divide the dataset into training, validation, and test sets to facilitate model evaluation and prevent overfitting.
# STEP 3:
Load the pretrained VGG19 model, initialized with ImageNet weights, to leverage its feature extraction capabilities.
# STEP 4:
Modify the architecture by removing the original fully connected layers and replacing the final output layer with a single neuron using a Sigmoid activation function for binary classification.
# STEP 5:
Train the model using the binary cross-entropy loss function and Adam optimizer, iterating through multiple epochs for optimal learning.
# STEP 6:
Assess performance by evaluating test data, analyzing key metrics such as the confusion matrix and classification report to measure accuracy and reliability in capacitor defect detection.

## PROGRAM
# Load Pretrained Model and Modify for Transfer Learning
```
from torchvision.models import VGG19_Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)
```
# Modify the final fully connected layer to match the dataset classes
```
num_classes=len(train_dataset.classes)
in_features=model.classifier[-1].in_features
model.classifier[-1]=nn.Linear(in_features,num_classes)
```
# Include the Loss function and optimizer
```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001
```
# Train the model
```
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:Dharanya        ")
    print("Register Number:212223230044       ")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/e941c887-5e06-414d-8df4-d28eab705430)
### Confusion Matrix
![image](https://github.com/user-attachments/assets/62b8153a-3d35-4be0-8a78-f8c5ddc35c58)
### Classification Report
![image](https://github.com/user-attachments/assets/adb1fdc2-249b-49e4-9bd0-771dcc03c079)
### New Sample Prediction
![image](https://github.com/user-attachments/assets/d072eda9-291a-4e57-bb85-c794a2311562)
![image](https://github.com/user-attachments/assets/de2f8eb2-04f5-41ec-bf4d-6eba4ab5fa63)
## RESULT
The VGG-19 transfer learning model was successfully implemented and trained. It achieved good classification performance with minimized training and validation losses, accurate predictions, and satisfactory evaluation metrics

