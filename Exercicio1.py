import numpy as np
import matplotlib.pyplot as plt

# Calcula o valor da função definida para a fonte pontual num dado instante t, para um
# determinado valor de x (convertido para um índice)- que deve ser comparado ao valor
# de x_{k} definido para o problema que se quer resolver -, de acordo com a fórmula
# dada no enunciado
def fonte(t, j, delta_x, c_square, beta=10, x_c=0.7):
    j_c = int(x_c/delta_x)

    if j != j_c:
        return 0.

    return 1000 * c_square * (1 - (2*(beta**2)*(t**2)*(np.pi**2))) * np.exp(-((beta**2)*(t**2)*(np.pi**2)))

# Utiliza a fórmula (10) do enunciado para calcular uma aproximação (discretizada) da solução
# da equação de onda u(x,t), armazenando os valores numa matriz - as linhas representam
# diferentes valores de t, e as colunas, diferentes valores de x (conforme a discretização adotada) -,
# a qual é dada como retorno da função
def metodo_explicito(n_t, n_x, delta_x, c_square, T=1):
    delta_t = T/n_t
    alpha = np.sqrt(c_square)*delta_t/delta_x
    u = np.zeros((n_t+1, n_x+1))

    for i in range(1, n_t):
        t_i = i*delta_t
        for j in range(1, n_x):
            u[i+1, j] = -u[i-1, j] + 2*(1-(alpha**2))*u[i, j] + (alpha**2)*(u[i, j+1] + u[i, j-1]) + \
                        (delta_t**2)*fonte(t_i, j, delta_x, c_square)

    return u

# Utiliza o pacote pyplot da biblioteca matplotlib para gerar um gráfico da solução do método explícito
# num dado instante t
def plot(u, t, n_t, n_x, delta_x, T=1):
    delta_t = T/n_t
    i = int(t/delta_t)
    x_axis = []

    for j in range(n_x+1):
        x_axis.append(j*delta_x)

    plt.plot(x_axis, u[i], 'r-')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Soluçao da equação de onda para um instante t=' + str(t) + ', com n_t=' + str(n_t))
    plt.savefig("solucao.png")
    plt.show()

def main():
    T = 1
    n_t = 350        # Pode ser alterado, caso desejado (usar valores inteiros)
    n_x = 100        # Pode ser alterado, caso desejado (usar valores inteiros)
    delta_x = 1/n_x
    c_square = 10    # Pode ser alterado, caso desejado (usar valores inteiros)
    t = 0.5          # Pode ser alterado, caso desejado

    while n_t != 300:
        u = metodo_explicito(n_t, n_x, delta_x, c_square, T)
        print('Solucao para n_t=' + str(n_t))
        plot(u, t, n_t, n_x, delta_x, T)
        n_t -= 10

# --------------------------------------------------------------------------
# Valores para o segundo item do exercício 1
#    n_t = 500
#    n_x = 100
#    delta_x = 1 / n_x
#    c_square = 20
#    t = 0.5
#
#    while n_t != 430:
#        u = metodo_explicito(n_t, n_x, delta_x, c_square, T)
#        print('Solucao para n_t=' + str(n_t))
#        plot(u, t, n_t, n_x, delta_x, T)
#        n_t -= 10
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Valores para plotagem dos gráficos
#    n_t = 1500
#    n_x = 200
#    delta_x = 1 / n_x
#    c_square = 20
#    t = 0.1
#
#     u = metodo_explicito(n_t, n_x, delta_x, c_square, T)
#     plot(u, t, n_t, n_x, delta_x, T)
# --------------------------------------------------------------------------

if __name__=="__main__":
    main()
