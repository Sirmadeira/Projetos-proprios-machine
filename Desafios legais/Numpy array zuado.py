import numpy as np
np.set_printoptions(threshold=np.inf)

def a():
    a=[]

    # Inicializando a matriz com 0 em todos as posições
    for i in range(200):
        j=a.append(list())
        for j in range(200):
            a[i].append(0)

    # Definindo numeros constantes
    a[0][0] = 9
    a[0][1] = -4
    a[0][2] = -1
    a[1][0] = -4
    a[1][1] = 6
    a[1][2] = -4
    a[1][3] = -1
    a[198][196] = 1
    a[198][197] = 4
    a[198][198] = -5
    a[198][199] = -2
    a[199][197] = 1
    a[199][198] = -2
    a[199][199] = 1

    # Gerando padrão diagonal
    for i in range(2, 198):
        a[i][i-2] = 1
        a[i][i-1] = -4
        a[i][i] = 6
        a[i][i+1] = -4
        a[i][i+2] = 1
    return np.array(a)

print(a())



