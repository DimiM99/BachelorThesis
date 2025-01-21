import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class SimpleNN_Torch(nn.Module):
    def __init__(self, input_size=784, hidden_size=10, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = F.relu(z1)
        z2 = self.fc2(a1)
        return F.softmax(z2, dim=1)

    def fit(self, X_train, Y_train, learning_rate=0.1, epochs=100):
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        X_tensor = torch.FloatTensor(X_train.T)  # Transpose to match PyTorch expected shape
        Y_tensor = torch.LongTensor(Y_train)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_tensor)
            loss = criterion(outputs, Y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                predictions = torch.argmax(outputs, 1)
                accuracy = (predictions == Y_tensor).sum().item() / Y_tensor.size(0)
                print(f"Epoch: {epoch}, Accuracy: {accuracy:.4f}")

    def evaluate(self, X, Y):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.T)
            Y_tensor = torch.LongTensor(Y)
            outputs = self(X_tensor)
            predictions = torch.argmax(outputs, 1)
            return (predictions == Y_tensor).sum().item() / Y_tensor.size(0)
