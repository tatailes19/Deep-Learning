import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


class CNN_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), stride=1),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 32, (3, 3), stride=1),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 128, (3, 3), stride=1),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)
    
    def compile(self, optimizer = optim.Adam,loss_fn = nn.CrossEntropyLoss(),**kwargs):
        self.optimizer = optimizer(self.parameters(), **kwargs)
        # if torch.cuda.is_available():
        #     self.clf = self.forward().to('cuda')
        # else:
        #     self.clf = self.forward().to("cpu")
            
        self.loss_fn = loss_fn
        
        
        

    def train(self,data_loader,epochs = 10):
        for epoch in range(epochs) :
            total_correct = 0
            total_samples = 0
            for batch in data_loader:
                X,y = batch
                if torch.cuda.is_available():
                    X,y = X.to('cuda'),y.to('cuda')
                else:
                    X,y = X.to('cpu'),y.to('cpu')
                yhat = self.forward(X)
                loss = self.loss_fn(yhat,y)
                predictions = torch.argmax(yhat, dim=1)
                total_correct += (predictions == y).sum().item()
                total_samples += y.size(0)
                 
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            accuracy = total_correct / total_samples
            print(f"Epoch:{epoch} loss is {loss.item()} train_accuracy is {accuracy}")
    
        # with open('model_state.pt', 'wb') as f: 
        #     save(clf.state_dict(), f)   
              
    def predict(self, data_loader): 
        predictions = []
        real = []
        for batch in data_loader:
            X, y = batch  
            yhat = self.forward(X)
            pred = torch.argmax(yhat, dim=1)
            predictions.extend(pred.cpu().numpy()) 
            real.append(int(y[0]))
        return predictions,real


