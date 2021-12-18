import torch



class Net(torch.nn.Module):
    def __init__(self, in_features=2, neurons=30):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, neurons)
        self.fc2 = torch.nn.Linear(neurons, neurons)
        self.fc3 = torch.nn.Linear(neurons, neurons)
        self.fc4 = torch.nn.Linear(neurons, 1)
        # self.relu = torch.nn.ReLU()
        self.relu = torch.nn.Tanh()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


if __name__ == "__main__":

    batch_size = 12
    in_features = 2

    x = torch.rand((batch_size, in_features))
    net = Net(in_features)

    out = net(x)

    print(x.shape)
    print( out )
    print( out.shape )
