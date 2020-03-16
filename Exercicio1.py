import numpy as np
import matplotlib.pyplot as plt

delta_x = 0.01
c_square = 10

def fonte(t, x):
    x_c = 0.7
    beta = 10

    if not np.isclose(x, x_c, atol=0.1):
        return 0

    return 1000*c_square*(1-2*((beta*t*np.pi())**2))*(np.e()**(-((beta*t*np.pi())**2)))

def metodo_explicito(n_t):
    T = 1
    n_x = 1/delta_x
    delta_t = T/n_t
    alpha = np.sqrt(c_square)*delta_t/delta_x
    u = np.empty([int(n_t+1), int(n_x+1)])

    for i in range(int(n_t+1)):
        t_i = i*delta_t
        for j in range(int(n_x+1)):
            if i == 0 or i == 1:
                u[i, j] = 0
            elif j == 0 or j == n_x:
                u[i+1, j] = 0
            else:
                x_j = j*delta_x
                u[i+1, j] = -u[i-1,j] + 2*(1-(alpha**2))*u[i,j] + (alpha**2)*(u[i,j+1] + u[i,j-1]) + (delta_t**2)*fonte(t_i, x_j)

    return u

def plot(u, t):
    n_x = 1/delta_x
    x_axis = []

    for j in range(int(n_x+1)):
        x_axis.append(j*delta_x)

    plt.plot(x_axis, u[t])
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Soluçao da equação de onda para um instante t=' + t)
    plt.savefig("solucao.png")
    plt.show()
