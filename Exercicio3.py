import numpy as np
from numpy import load
import matplotlib.pyplot as plt

# Calcula o valor da função definida para a fonte pontual num dado instante t, para um
# determinado valor de x (convertido para um índice) - que deve ser comparado ao valor
# de x_{k} definido para o problema que se quer resolver -, de acordo com a fórmula
# dada no enunciado
def fonte(t, j, x_k, delta_x, c_square, beta=10):
    j_k = int(x_k/delta_x)

    if j != j_k:
        return 0.

    return 1000 * c_square * (1 - (2*(beta**2)*(t**2)*(np.pi**2))) * np.exp(-((beta**2)*(t**2)*(np.pi**2)))

# Utiliza a fórmula (10) do enunciado para calcular uma aproximação (discretizada) da solução
# da equação de onda u(x,t), armazenando os valores numa matriz - as linhas representam
# diferentes valores de t, e as colunas, diferentes valores de x (conforme a discretização adotada) -,
# a qual é dada como retorno da função
def metodo_explicito(n_t, n_x, c_square, x_k, delta_t, delta_x=0.01):
    alpha = np.sqrt(c_square)*delta_t/delta_x
    u = np.zeros((n_t + 1, n_x + 1))

    for i in range(1, n_t):
        t_i = i * delta_t
        for j in range(1, n_x):
            u[i+1, j] = -u[i-1, j] + 2*(1-(alpha**2))*u[i, j] + (alpha**2)*(u[i, j+1] + u[i, j-1]) + \
                        (delta_t**2)*fonte(t_i, j, x_k, delta_x, c_square)

    return u

# Função auxiliar responsável por calcular o produto de uma matriz com um vetor
def produto_matrizes(a, b):
    r_a, c_a = a.shape
    r_b = len(b)

    if c_a != r_b:
        print('O produto das matrizes passadas como parametro nao esta definido\n')
        return None

    return [sum(x*y for x,y in zip(a_row,b)) for a_row in a]

# Função que implementa a fórmula dos trapézios dada no enunciado para realizar a
# a aproximação numérica de uma integral num dado intervalo
def integral(g, delta_t, n):
    soma = 0.0

    for i in range(1, n):
        soma += g[i]
    return delta_t / 2 * (g[0] + g[n] + 2*soma)

# Calcula o produto escalar usado para construção do sistema normal (matrizes B e c)
# utilizado para determinar os parâmetros a_{0},...,a_{k}, tendo em vista que a matriz
# B do sistema normal do MMQ no caso contínuo é uma matriz simétrica cujos elementos
# correspondem ao produto escalar - definido como a integral do produto de duas funções
# num determinado intervalo de integração - entre duas funções u_{i} e u_{j} calculadas
# com o método explícito, e que a matriz c tem seus elementos dados pelo produto escalar
# entre a função conhecida d_{r} e as funções u_{i} determinadas anteriormente
def produto_escalar(f, g, delta_t, n):
    produto = [x * y for x,y in zip(f, g)]
    return integral(produto, delta_t, n)

# Aplica um fator de ruído aos dados (dr) de acordo com a fórmula dada no enunciado
def d_ruido(dr, eta, n):
    delta = eta * max(abs(ele) for ele in dr)

    d_ruidoso = np.zeros(n+1)
    for i in range(n+1):
        v = 1 - np.random.rand()*2 # Gera um número aleatório entre -1 e 1
        d_ruidoso[i] = (1 + delta * v) * dr[i]

    return d_ruidoso

# Calcula o nível de ruído (em porcentagem) através da fórmula dada no enunciado
def nivel_ruido(dr, d_ruidoso, n_t, t_i, t_f, T=1):
    delta_t = T / n_t
    n = int((t_f - t_i) / delta_t)

    numerador = [abs(x - y) for x,y in zip(dr, d_ruidoso)]
    denominador = [abs(x) for x in dr]

    return 100 * (integral(numerador, delta_t, n)) / (integral(denominador, delta_t, n))

# Usando a função de produto escalar definida acima, constrói as matrizes B e c do sistema
# normal a ser resolvido, tendo como retorno, além dessas matrizes, um vetor u que contém
# as funções u_{k} (listas) para um determinado x_{r}, um vetor d que contém os valores
# do vetor dr (passado como parâmetro) entre os intervalos ti e tf (convertidos para índi-
# ces do vetor, conforme a discretização adotada), e um vetor d_ruidoso que contém os valores
# do vetor d (calculado na função) submetidos a um fator de ruído - sendo que estes três úl-
# timos retornos são usados apenas para os cálculos do erro, do resíduo e do nível de ruído
def sistema_normal(K, n_t, c_square, t_i, t_f, x_k, dr, eta, x_r=0.7, delta_x=0.01, T=1):
    delta_t = T / n_t
    n_x = int(1/delta_x)
    i_ini = int(t_i / delta_t)
    n = int((t_f - t_i) / delta_t)
    j = int(x_r / delta_x)

    # Costrói o vetor u_{k} no intervalo [ti, tf] a partir da matriz u obtida
    # por meio do método explícito para um dado k e um instante x_{k}. O vetor
    # u_{k} tem K linhas e (n+1) colunas (representam os dados no intervalo escolhido)
    u_k = np.zeros((K, n + 1))
    for k in range(K):
        u = metodo_explicito(n_t, n_x, c_square, x_k[k], delta_t, delta_x)
        for i in range(n + 1):
            u_k[k, i] = u[i_ini + i, j]

    # Cria um vetor d que contém os valores do parêmtro dr no intervalo [ti, tf]
    # e um vetor d_ruidoso com os dados ruidosos
    d = np.zeros(n + 1)
    for i in range(n + 1):
        d[i] = dr[i_ini + i]
    d_ruidoso = d_ruido(d, eta, n)  # Adiciona o fator de ruido

    B = np.zeros((K, K))
    c = []

    for k in range(K):
        c.append(produto_escalar(u_k[k], d_ruidoso, delta_t, n))
        B[k, k] = produto_escalar(u_k[k], u_k[k], delta_t, n)
        for j in range(k):
            B[k, j] = B[j, k] = produto_escalar(u_k[j], u_k[k], delta_t, n)

    return B, c, u_k, d, d_ruidoso

# Realiza a fatoração de Cholesky transformando a matriz B do sistema normal em
# matrizes triangulares inferior e superior (conforme algoritmo dado em sala de
# aula), e resolve o sistema correspondente usando substituição para frente (com
# a matriz triangular inferior), e substituição retroativa (para a matriz triangular
# superior), tendo como retorno um vetor com a solução (parâmetros a_{k})
def fatoracao_Cholesky(B, c):
    L = np.zeros_like(B)
    n,_ = B.shape

    # Monta a matriz triangular inferior (pelo algoritmo dado em sala de aula)
    L[0, 0] = np.sqrt(B[0, 0])
    for i in range(1, n):
        for j in range(i):
            temp_sum = sum((L[i, k] * L[j, k]) for k in range(j))
            L[i, j] = (B[i, j] - temp_sum) / L[j, j]
        L[i, i] = np.sqrt(B[i, i] - sum((L[i, k]**2) for k in range(i)))

    # Monta a matriz triangular superior (transposta de L)
    L_T = np.zeros_like(B)
    for i in range(n):
        for j in range(n):
            L_T[i, j] = L[j, i]

    # Substituição para frente
    y = np.zeros(n)
    for i in range(n):
        temp_sum = 0.0
        for j in range(i):
            temp_sum += L[i, j] * y[j]
        y[i] = (c[i] - temp_sum) / L[i, i]

    # Substituição retroativa
    a = np.zeros(n)
    for i in range(n-1, -1, -1):
        temp_sum = 0
        for j in range(i+1, n):
            temp_sum += L_T[i][j] * a[j]
        a[i] = (y[i] - temp_sum) / L_T[i][i]

    return a

# Função auxiliar que calcula a norma dos elementos de um vetor (raiz
# quadrada da soma dos quadrados dos elementos), usada para checar o critério
# de parada do método SOR
def norma(A):
    norm = 0
    n = len(A)

    for i in range(n):
        norm += abs(A[i])**2

    return np.sqrt(norm)

# Para matrizes B e c do sistema que se quer resolver, implementa o método de
# SOR visto em aula, tendo como parâmetros um certo omega, um vetor de valores
# iniciais (a partir do qual se faz as iterações), um número máximo de iterações
# e uma tolerância usada como critério de parada.
# Nesse sentido, o método calcula novas iterações para o vetor solução a até que
# se atinja o número máximo de iterações, ou até que a norma do resultado do produto
# da matriz B com o vetor solução a atual subtraído do vetor c seja menor que a tole-
# rância definida para o algoritmo, retornando a solução encontrada
def metodo_SOR(B, c, omega, a0, itmax=10000, tol=1e-16):
    K = len(c)

    a = np.array(a0)
    for k in range(1, itmax):
        for i in range(K):
            sigma = 0
            for j in range(K):
                if j != i:
                    sigma += B[i, j] * a[j]
            a[i] = (1 - omega) * a[i] + (omega / B[i, i]) * (c[i] - sigma)
        # Critério de parada: verifica se a norma da matriz resultante da subtração
        # do produto das matrizes a (calculada iterativamente) e B e da matriz c
        # é menor que a tolerância definida
        if norma([x-y for x,y in zip(produto_matrizes(B, a),c)]) < tol:
            break

    return a

# Função auxiliar que calcula o erro do algoritmo implementado (conforme
# definição do enunciado) e imprime o resultado
def print_erro(a_barra, a_estrela):
    erro = 0
    for k in range(len(a_barra)):
        erro += (a_barra[k] - a_estrela[k])**2

    erro = np.sqrt(erro)
    print(erro)

# Função auxiliar que calcula o resíduo do algoritmo implementado (conforme
# definição (27) do enunciado) e imprime o resultado
def print_residuo(K, u, a, dr, n_t, t_i, t_f, T=1):
    delta_t = T / n_t
    n = int((t_f - t_i) / delta_t)

    soma = np.zeros(n+1)
    for i in range(n+1):
        for k in range(K):
            soma[i] += a[k]*u[k, i]

    integrando = [((x - y)**2) for x,y in zip(soma, dr)]
    res = integral(integrando, delta_t, n) / (2 * (t_f - t_i))
    print(res)

# Utiliza o pacote pyplot da biblioteca matplotlib para gerar um gráfico com os dados com e sem
# ruído num intervalo [ti, tf]
def plot(n_t, dr, d_ruidoso, t_i, t_f, eta, T=1):
    delta_t = T / n_t
    n = int((t_f - t_i) / delta_t)

    t = [t_i + i * delta_t for i in range(n+1)]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax1.plot(t, dr, '-r')
    ax1.set_xlabel('t')
    ax1.set_ylabel('Medição (d_r)')
    ax1.set_title('Sismograma sem ruído com eta=' + str(eta))
    ax2.plot(t, d_ruidoso, '-r')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Medição (d_r)')
    ax2.set_title('Sismograma com ruído com eta=' + str(eta))
    plt.savefig("sismograma.png")
    plt.show()

def main():
    T = 1
    n_t = 1000                                                          # Pode ser alterado, caso desejado (usar valores inteiros)
    c_square = 20                                                       # Pode ser alterado, caso desejado (usar valores inteiros)
    delta_x = 0.01                                                      # Pode ser alterado, caso desejado
    x_r = 0.7
    x_k = [0.03, 0.15, 0.17, 0.25, 0.33, 0.34, 0.40, 0.44, 0.51, 0.73]  # Pode ser alterado, caso desejado
    t_i = 0.9                                                           # Pode ser alterado, caso desejado
    t_f = 1.0                                                           # Pode ser alterado, caso desejado
    eta = 1e-03                                                         # Pode ser alterado, caso desejado (usar 1e-3, 1e-4 ou 1e-5)
    K = 10                                                              # Pode ser alterado, caso desejado (usar 3, 10 ou 20)
    dr = load('dr' + str(K) + '.npy')

    B, c, u, d, d_ruidoso = sistema_normal(K, n_t, c_square, t_i, t_f, x_k, dr, eta, x_r, delta_x, T)
    a = fatoracao_Cholesky(B, c)
    print('Solucao obtida pela fatoracao de Cholesky (dados ruidosos):')
    print(a)
    a_estrela = [7.3, 2.4, 5.7, 4.7, 0.1, 20.0, 5.1, 6.1, 2.8, 15.3]  # Pode ser alterado, caso desejado (valores da solução conhecida)
    print('Solucao exata conhecida:')
    print(a_estrela)
    print('Erro da fatoracao de Cholesky:')
    print_erro(a, a_estrela)
    print('Residuo da fatoracao de Cholesky:')
    print_residuo(K, u, a, d, n_t, t_i, t_f, T)
    print('Nivel de ruido:')
    print('{:.3f}'.format(nivel_ruido(d, d_ruidoso, n_t, t_i, t_f, T)) + '%')
    plot(n_t, d, d_ruidoso, t_i, t_f, eta, T)

# --------------------------------------------------------------------------
# Valores para o método SOR
#    omega = 1.6
#   a0 = np.zeros(K)
#    a = metodo_SOR(B, c, omega, a0, itmax=10000)
#    print('\nSolucao obtida pelo metodo SOR:')
#    print(a)
#    print('Solucao exata conhecida:')
#    print(a_estrela)
#    print('Erro do metodo SOR:')
#    print_erro(a, a_estrela)
#    print('Residuo do metodo SOR:')
#    print_residuo(K, u, a, d, n_t, t_i, t_f, T)
#    print('Nivel de ruido:')
#    print('{:.3f}'.format(nivel_ruido(d, d_ruidoso, n_t, t_i, t_f, T)) + '%')
#    plot(n_t, d, d_ruidoso, t_i, t_f, eta, T)
# --------------------------------------------------------------------------

if __name__=="__main__":
    main()