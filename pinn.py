# Solve the Burgers' equation using neural network
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


import network


def np2torch(dat):
    return torch.from_numpy(dat.astype(np.float32)).clone()


def data_initial_boundary(n_inital, n_boundary):
    tx_initial, u_initial = initial_condition(n_inital)
    tx_boundary, u_boundary = boundary_condition(n_boundary)
    tx = np.vstack((tx_initial, tx_boundary))
    u = np.hstack((u_initial, u_boundary))
    u = np2torch(u)
    u = torch.unsqueeze(u, 1)
    return np2torch(tx), u


def initial_condition(num):
    '''
    u(0,x) = -sin( pi * x )
    '''
    t = np.zeros(num)
    x = np.random.uniform(low=-1.0, high=1.0, size=num)
    u = -np.sin(np.pi * x)
    data = np.dstack((t, x))[0]  # t, x : shape = (num, 2)
    # print(data)
    return data, u


def boundary_condition(num):
    '''
    u(t,-1) = u(t,1) = 0
    '''
    t = np.random.uniform(low=0.0, high=1.0, size=num)
    x0 = np.full(num, -1.0)
    x1 = np.full(num,  1.0)
    u  = np.full(num*2,  0.0)
    data0 = np.dstack((t, x0))[0]
    data1 = np.dstack((t, x1))[0]
    data = np.vstack((data0, data1))
    # print(data)
    return data, u


def collocation_points(num):
    '''
    PDE
    '''
    t = torch.rand((num, 1), requires_grad=True)
    x = np.random.uniform(low=-1.0, high=1.0, size=num)
    x = x.reshape([num, 1])
    x = np2torch(x)
    x.requires_grad = True
    tx = torch.hstack((t, x))
    return t, x, tx


def train(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)

    # initial & boundary condition data
    tx_input, u_target = data_initial_boundary(400, 50)
    tx_input = tx_input.to(device)
    u_target = u_target.to(device)
    
    # PDE data
    n_f = 10000
    t_f, x_f, tx_f = collocation_points(n_f)
    t_f = t_f.to(device)
    x_f = x_f.to(device)
    tx_f = tx_f.to(device)


    net = network.Net().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.LBFGS(net.parameters(), max_iter=epochs)

    net.train()

    loss_history = []
    def closure():
        optimizer.zero_grad()

        u_pred = net(tx_input)
        loss_ib = criterion(u_pred, u_target)


        u = net(tx_f)
        u_x = torch.autograd.grad(u, x_f, create_graph=True, grad_outputs=torch.ones_like(u), allow_unused=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_f, create_graph=True, grad_outputs=torch.ones_like(u), allow_unused=True, retain_graph=True)[0]
        u_t = torch.autograd.grad(u, t_f, create_graph=True, grad_outputs=torch.ones_like(u), allow_unused=True, retain_graph=True)[0]
        
        if u_x is None:
            u_x = torch.zeros_like(u)
        if u_xx is None:
            u_xx = torch.zeros_like(u)
        if u_t is None:
            u_t = torch.zeros_like(u)

        nu = 0.01/np.pi
        f = u_t  +  u * u_x  -  nu * u_xx
        loss_f = criterion(f, torch.zeros_like(f))


        loss = loss_ib + loss_f
        loss.backward()
        loss_history.append(loss.item())
        return loss

    optimizer.step(closure)

    torch.save(net.state_dict(), "pinn_burgers.pt")

    plot_u(net)
    plot_loss(loss_history)
    plot_points(tx_f)


def plot_u(net):
    x = torch.linspace(-1, 1, steps=360)
    net.eval()
    with torch.no_grad():
        fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        axs[0].set_ylabel('u [m/s]')
        for i, tt in enumerate((0.25, 0.5, 0.75)):
            t = torch.full_like(x, tt)
            tx = torch.dstack((t, x))[0]
            u_pred = net(tx)
            axs[i].plot(x, u_pred, linewidth=2.0)
            axs[i].set_xlim(-1, 1)
            axs[i].set_ylim(-1.5, 1.5)
            axs[i].set_title('t = ' + str(tt) + 'sec')
            axs[i].set_xlabel('x [m]')
            
        plt.show()

def plot_loss(loss_history):
    fig, ax = plt.subplots()
    ax.plot(loss_history)
    # ax.set_xlim(0, len(loss_history))
    ax.set_title("Loss history")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_yscale('log')
    plt.show()


def plot_points(tx):
    fig, ax = plt.subplots()

    tx = tx.to('cpu').detach().numpy().copy()

    x = tx[:, 0]
    y = tx[:, 1]

    print(tx.shape)
    print(type(tx))
    ax.scatter(x, y, marker='x')

    ax.set_xlabel('time [sec]')
    ax.set_ylabel('x [m]')
    plt.show()



if __name__ == '__main__':
    train(100)