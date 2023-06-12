import torch
from torch import nn

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        self.model = nn.Sequential(nn.Linear(2, 8),nn.ReLU(),nn.Linear(8, 1))
        
    def forward(self, x) -> torch.Tensor:
        return self.model(x)
    
    def train(self, x_train, y_train, x_test, y_test, num_epochs, optim, loss_fun, lr) -> dict:
        loss_fn: nn.CrossEntropyLoss = loss_fun()
        optimizer: torch.optim.SGD = optim(self.parameters(), lr=lr)

        losses = list()
        test_losses = list()

        for epoch in range(num_epochs):
            # forward pass
            optimizer.zero_grad()
            preds = self.forward(x_train)

            # calculate loss
            loss: torch.Tensor = loss_fn(preds, y_train)
            loss.backward()

            losses.append(loss.item())

            # updating weights
            optimizer.step()

            # calculating test loss
            with torch.inference_mode():
                test_preds = self.forward(x_test)
                test_loss: torch.Tensor = loss_fn(test_preds, y_test)

                test_losses.append(test_loss.item())

        return {"loss": losses, "test_loss": test_losses}
            