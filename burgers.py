import numpy as np
import matplotlib.pyplot as plt


def burgers(nt, nx):
    '''
    FTCS method
    '''
    nu = 0.01/np.pi
    nu = 0.006
   

    x_min = -1
    x_max = 1

    t_min = 0
    t_max = 1

    u = np.zeros((nt, nx))
    # print(u.shape)

    x = np.linspace(x_min, x_max, nx)
    # print(x)

    dt = (t_max - t_min) / (nt-1)
    dx = (x_max - x_min) / (nx-1)
    # print(dt)
    # print(dx)

    # Initial condition
    u[0] = -np.sin(np.pi*x)

    for n in range(0, nt-1):
        for i in range(1, nx-1):
            u_diff1 = ( u[n][i+1] - u[n][i-1] )/(2*dx)
            u_diff2 = ( u[n][i+1] - 2*u[n][i] + u[n][i-1]) /(dx*dx)
            u[n+1][i] = u[n][i] - dt*( u[n][i] * u_diff1 - nu * u_diff2 )
        u[n+1][0] = 0
        u[n+1][-1] = 0


    # print("---")
    # print(u)

    u_max = np.amax(u)
    print(u_max)

    courant = u_max * dt / dx
    print(courant)
    return u




def draw_contour(u):
    nt = u.shape[0]
    nx = u.shape[1]

    t = np.linspace( 0, 1, nt)
    x = np.linspace(-1, 1, nx)
    T, X = np.meshgrid(t, x)

    plt.pcolormesh(T, X, u.T, cmap='rainbow', shading='auto') # 等高線図の生成

    cb = plt.colorbar (orientation="vertical") # カラーバーの表示 
    cb.set_label("u", fontname="Arial", fontsize=12) #カラーバーのラベル

    plt.xlabel('time', fontsize=12)
    plt.ylabel('x', fontsize=12)
    plt.show()



if __name__ == "__main__":
    nt = 2000
    nx = 200
    u = burgers(nt, nx)
    draw_contour(u)