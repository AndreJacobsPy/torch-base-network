import torch
from torch import nn

class BaseNet(nn.Module):
    """
    This is a class that helps to make building Neural Networks with PyTorch easier.

    Methods:
        train:
            arguments:
                xtrain: this is the inputs of model. must be in torch.Tensor dtype.
                ytrain: this is the outputs of model. must be in torch.Tensor dtype.
                xtest: this is inputs to test model. must be in torch.Tensor dtype.
                ytest: this is outputs to test model. must be in torch.Tensor dtype.
                num_epochs: how many iterations to train the model
                optim: the torch optimizer to use. typically Adam is the best.
                loss_fun: the loss function to minimize while training
                lr: the learning rate of the model
    """

    def __init__(self):
        super(BaseNet, self).__init__()

        self.model = nn.Sequential(nn.Linear(2, 8),nn.ReLU(),nn.Linear(8, 1))
        
    def forward(self, x) -> torch.Tensor:
        return self.model(x)
    
    def train(self, x_train, y_train, x_test, y_test, num_epochs, optim, loss_fun, lr) -> dict:
        """
        arguments:
                xtrain: this is the inputs of model. must be in torch.Tensor dtype.
                ytrain: this is the outputs of model. must be in torch.Tensor dtype.
                xtest: this is inputs to test model. must be in torch.Tensor dtype.
                ytest: this is outputs to test model. must be in torch.Tensor dtype.
                num_epochs: how many iterations to train the model
                optim: the torch optimizer to use. typically Adam is the best.
                loss_fun: the loss function to minimize while training
                lr: the learning rate of the model

        example:
            # creating some fake data
            xtr = torch.Tensor([[1, 2],
                                [3, 4],
                                [5, 6]])
            ytr = torch.Tensor([[1.5],
                                [3.5],
                                [5.5]])

            xte = torch.Tensor([[3, 2],
                                [5, 1]])
            yte = torch.Tensor([[2.5],
                                [3.5]])

            # build network
            class MyNet(BaseNet):
                def __init__(self):
                    super(MyNet, self).__init__()

                    self.model = torch.nn.Sequential(
                        torch.nn.Linear(2, 8),
                        torch.nn.ReLU(),
                        torch.nn.Linear(8, 1)
                    )

            # train network
            model = MyNet()
            output = model.train(xtr, ytr, xte, yte, 5, torch.optim.Adam, torch.nn.MSELoss, 0.001)
            print(output)
        """

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
            