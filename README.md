# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
The task is to build a Convolutional Neural Network (CNN) model that can automatically classify images into different categories. Instead of manually identifying images, the model should learn patterns from the data and predict the correct class.

### Fashion-MNIST dataset:

60,000 training images
10,000 testing images
10 categories of clothes
Image size: 28 × 28 (grayscale)

## Neural Network Model

![DLExp3](https://github.com/user-attachments/assets/a213c96b-9ea3-4477-954b-e07376ee8ec5)



## DESIGN STEPS
### STEP 1: 

Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

### STEP 2: 

Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

### STEP 3: 

Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4: 

Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

### STEP 5: 

Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

### STEP 6: 

Evaluate the trained model on test images and verify the classification accuracy for new unseen images.



## PROGRAM

### Name: ASTLE JOE A S

### Register Number: 212224240019

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32, kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1=nn. Linear (128*3*3,128)
        self.fc2=nn. Linear(128,64)
        self.fc3 = nn.Linear(64, 10)   # assuming 10 output classes


    def forward(self, x):
        # write your code here
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)

# Train the Model
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
      optimizer.zero_grad()
      outputs=model(images)
      loss= criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print('Name: ASTLE JOE A S')
    print('Register Number:212224240019')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


```

### OUTPUT

## Load Fashion-MNIST dataset

<img width="469" height="84" alt="image" src="https://github.com/user-attachments/assets/5b83e042-9109-484c-8aa8-a06d3d6af32a" />

<img width="575" height="471" alt="image" src="https://github.com/user-attachments/assets/6869c5b1-7386-4d48-abe5-6fce93a5d7fc" />


## Training Loss per Epoch

<img width="317" height="208" alt="image" src="https://github.com/user-attachments/assets/76d525ac-9734-49d6-b4a3-d8a5910a4b7b" />


## Confusion Matrix

<img width="806" height="787" alt="image" src="https://github.com/user-attachments/assets/7baf0b9c-d395-4c03-9c80-8e2432f80bb9" />


## Classification Report

<img width="483" height="399" alt="image" src="https://github.com/user-attachments/assets/603cea4d-d2b1-44f4-9cf4-a3586675729a" />


### New Sample Data Prediction

<img width="454" height="556" alt="image" src="https://github.com/user-attachments/assets/984ca851-9103-438c-b80b-b464981b28ac" />


## RESULT

The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.
