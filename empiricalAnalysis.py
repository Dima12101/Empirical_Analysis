import matplotlib.pyplot as plt
import numpy as np

from algorithmDijkstra import Algorithm_Dijkstra

RANGE_n = (1, 101)


def _show_empirical_f_with_asymptotic(empirical_f, C1, C2):
    x = np.arange(*RANGE_n)
    lowerAsymptotic = C1 * x ** 2
    upperAsymptotic = C2 * x ** 2
    fig, ax = plt.subplots()
    ax.plot(x, upperAsymptotic, color="green", linestyle='--', label="$C_2n^2$")
    ax.scatter(x, empirical_f, marker='o', s=10, c="red", edgecolor='b', label="$f(n)$")
    ax.plot(x, lowerAsymptotic, color="orange", linestyle='--', label="$C_1n^2$")
    ax.set_xlabel("n")
    ax.set_ylabel("Трудоёмкость")
    ax.minorticks_on()
    ax.grid(which='major',
            color='k',
            linestyle=':')
    ax.grid(which='minor',
            color='k',
            linestyle=':')
    ax.legend()

    plt.show()
    fig.savefig('empiricalAnalysis_data/empirical_analysis.png')


def _show_empirical_f(empirical_f):
    x = np.arange(*RANGE_n)
    fig, ax = plt.subplots()
    ax.scatter(x, empirical_f, marker='o', s=10, c="red", edgecolor='b', label="$f(n)$")
    ax.set_xlabel("n")
    ax.set_ylabel("Трудоёмкость")
    ax.minorticks_on()
    ax.grid(which='major',
            color='k',
            linestyle=':')
    ax.grid(which='minor',
            color='k',
            linestyle=':')
    ax.legend()

    plt.show()
    fig.savefig('empiricalAnalysis_data/empirical_f_test1.png')


def empirical_analysis():
    Dijkstra = Algorithm_Dijkstra()
    range_n = RANGE_n
    m = 100
    repeats = 100

    '''Getting of empirical f'''
    f = [None] * (range_n[1] - range_n[0])
    for i, n in enumerate(range(*range_n)):
        print(n)
        f_on_n = [None] * m
        for j in range(m):
            Graph, start_node = Dijkstra.generate_data(n)
            f_repeats = [0] * repeats
            for l in range(repeats):
                _, _, f_repeats[l] = Dijkstra.algorithm(Graph, start_node)
            f_on_n[j] = sum(f_repeats) / repeats
        f[i] = sum(f_on_n) / m

    '''Up/down load data'''
    # with open('empiricalAnalysis_data/empirical_f_data.txt', 'w') as data:
    #     for f_i in f:
    #         data.write(f"{str(f_i)}\n")
    # with open('empiricalAnalysis_data/empirical_f_data.txt', 'r') as data:
    #     for i in range(len(f)):
    #         f[i] = float(data.readline())

    _show_empirical_f(f)

    '''Search constants: C1 and C2'''
    n = np.arange(*range_n)
    f = np.array(f)
    g = n ** 2
    ratio_f_g = f / g
    for i in range(len(n)):
        C1 = min(ratio_f_g[i:])
        C2 = max(ratio_f_g[i:])
        if C1 > 0 and C2 > 0:
            print('n0:', n[i])
            print('C1:', C1)
            print('C2:', C2)
            break
    # C1 = 0.00020716132941069426 C2 = 0.0002993999481201172
    _show_empirical_f_with_asymptotic(f, C1, C2)

    '''Ratio: f(n)/n^2'''
    #asymptotic
    C = min(ratio_f_g)
    print('C', C)
    
    fig, ax = plt.subplots()
    ax.set_xlabel("n")
    ax.scatter(range(*range_n), ratio_f_g, marker='o', s=4, c="red", edgecolor='b')
    ax.plot(range(*range_n), ratio_f_g, color="red", linestyle='--', label="$f(n)/n^2$")
    ax.plot(range(*range_n), [C] * (range_n[1] - range_n[0]), color="green", linestyle='--', label="C")
    ax.legend()
    plt.show()
    fig.savefig('empiricalAnalysis_data/ratio_f_g.png')

    '''Ratio: f(2n)/f(n)'''
    n = range(*range_n)[:range_n[1] // 2 - range_n[0] + 1]
    f_n_1 = np.array(f[:range_n[1] // 2 - range_n[0] + 1])
    f_n_2 = np.array(f[range_n[0] * 2 - range_n[0]::2])
    ratio_f2_f1 = f_n_2 / f_n_1
    fig, ax = plt.subplots()
    ax.set_xlabel("n")
    ax.scatter(n, ratio_f2_f1, marker='o', s=4, c="red", edgecolor='b')
    ax.plot(n, ratio_f2_f1, color="red", linestyle='--', label="$f(2n)/f(n)$")
    ax.legend()
    plt.show()
    fig.savefig('empiricalAnalysis_data/ratio_f2_f1.png')


