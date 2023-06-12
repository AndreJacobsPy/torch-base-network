from tbn.network import BaseNet
import torch

if __name__ == "__main__":
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
