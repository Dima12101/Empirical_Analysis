import sys
import random
import matplotlib.pyplot as plt
from scipy import stats, special
import numpy as np
import math
import time
import heapq


class Algorithm_Dijkstra:
    def __init__(self):
        self.RANGE_VALUES = (1, 10)
        self.MAX_DISTANCE = sys.maxsize

    def _generate_graph(self, count_V, count_E):
        # Генерация не связанного графа
        graph = list([0] * count_V for i in range(0, count_V))
        # Добавление рёбер
        temp_count_E = 0
        while temp_count_E < count_E:
            # Случайным образом выбираем пару вершин
            begin_node = random.randint(0, count_V - 1)
            end_node = random.randint(0, count_V - 1)
            if graph[begin_node][end_node] == 0:
                # Случайным образом выясняем, есть ли между ними ребро
                is_exist_E = random.randint(False, True)
                if is_exist_E:
                    # Создаём ребро
                    graph[begin_node][end_node] = random.randint(*self.RANGE_VALUES)
                    temp_count_E += 1
        return graph

    def generate_data(self, n):
        # Случайным образом выбираем кол-во рёбер
        count_E = random.randint(0, n * (n + 1) / 2)
        # Случайным образом генерируем граф
        Graph = self._generate_graph(count_V=n, count_E=count_E)
        # Случайным образом выбираем стартовую вершину
        start_node = random.randint(0, n - 1)

        return Graph, start_node

    def algorithm(self, Graph, start_node):
        # Инициализация алгоритма
        n = len(Graph)

        parent = [None] * n
        distance = [self.MAX_DISTANCE] * n
        distance[start_node] = 0

        Q = []
        for i in range(n):
            heapq.heappush(Q, (distance[i], i))

        V = set(range(n))
        V_T = set({})

        time_start = time.time()
        while len(Q) != 0:
            min_node = heapq.heappop(Q)[1]
            V_T.add(min_node)
            for node in V - V_T:
                if Graph[min_node][node]:
                    new_distance = distance[min_node] + Graph[min_node][node]
                    if distance[node] > new_distance:
                        Q.remove((distance[node], node))
                        heapq.heappush(Q, (new_distance, node))

                        distance[node] = new_distance
                        parent[node] = min_node

        time_end = time.time()
        # В миллисекундах
        alg_time = (time_end - time_start) * 1000
        return distance, parent, alg_time


def print_result(distance, start_node):
    print("Стоимость пути из начальной вершины до остальных:")
    for i in range(0, len(distance)):
        if distance[i] != sys.maxsize:
            print(f"{start_node} > {i} = {distance[i]}")
        else:
            print(f"{start_node} > {i} = маршрут недоступен")


# Empirical analysis -----------------------------------------
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
    fig.savefig('empirical_analysis.png')


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
    fig.savefig('empirical_f.png')


def empirical_analysis():
    Dijkstra = Algorithm_Dijkstra()
    range_n = RANGE_n
    m = 100

    '''Getting of empirical f'''
    f = [None] * (range_n[1] - range_n[0])
    for i, n in enumerate(range(*range_n)):
        f_on_n = [None] * m
        for j in range(m):
            Graph, start_node = Dijkstra.generate_data(n)
            _, _, f_on_n[j] = Dijkstra.algorithm(Graph, start_node)
        f[i] = sum(f_on_n) / m
    _show_empirical_f(f)

    '''For showing of table'''
    with open('data.txt', 'w') as data:
        for n in range(*range_n):
            data.write(f"{n}\t")
        data.write('\n')
        for f_i in f:
            data.write(f"{str(f_i).replace('.',',')}\t")

    '''Search constants: C1 and C2'''
    n = np.arange(*range_n)
    f = np.array(f)
    g = n ** 2
    ratio = f / g
    for i in range(len(n)):
        C1 = min(ratio[i:])
        C2 = max(ratio[i:])
        if C1 > 0 and C2 > 0:
            print('n0:', n[i])
            print('C1:', C1)
            print('C2:', C2)
            _show_empirical_f_with_asymptotic(f, C1, C2)
            return True
    return False

# -------------------------------------------------------------


def _beta_distribution(alpha, beta):
    beta_coef = special.gamma(alpha + beta) / (special.gamma(alpha) * special.gamma(beta))
    return lambda x: beta_coef * math.pow(x, alpha - 1) * math.pow(1 - x, beta - 1)


def checking_distribution():
    Dijkstra = Algorithm_Dijkstra()

    '''Parameters'''
    n = 50
    m = 20000
    repeats = 100

    '''Counting of values f for the fixed n'''
    f_n = [0] * m
    # for i in range(m):
    #     Graph, start_node = Dijkstra.generate_data(n)
    #     f_repeats = [0] * repeats
    #     for j in range(repeats):
    #         _, _, f_repeats[j] = Dijkstra.algorithm(Graph, start_node)
    #     f_n[i] = sum(f_repeats) / repeats

    '''Number of segments'''
    #k = math.floor(1 + math.log2(m))
    k = math.floor(math.pow(m, 1/3))

    '''------Temp------'''
    # with open('f-values.txt', 'w') as f_file:
    #     for f in f_n:
    #         f_file.write(str(f) + '\n')
    with open('f-values.txt', 'r') as f_file:
        for i, line in enumerate(f_file.readlines()):
            f_n[i] = float(line)
    '''---------------'''

    '''Histogram of f'''
    # fig, ax = plt.subplots()
    # w, bins, _ = ax.hist(f_n, bins=k, normed=True)
    # plt.show()
    # fig.savefig('hist-f.png')

    '''Some parameters'''
    f_n_mean = sum(f_n) / m
    f_n_min = min(f_n)
    f_n_max = max(f_n)

    '''Normalization f --> t'''
    t = (np.array(f_n) - f_n_min) / (f_n_max - f_n_min)
    '''Histogram of t'''
    # fig, ax = plt.subplots()
    # ax.hist(t, bins=k, normed=True)
    # plt.show()
    # fig.savefig('hist-t.png')

    '''Counting of ALPHA and BETTA for beta-distribution'''
    t_mean = (f_n_mean - f_n_min) / (f_n_max - f_n_min)
    s_2 = sum(((np.array(f_n) - f_n_mean) ** 2) / ((f_n_max - f_n_min) ** 2)) / (m - 1)
    a = (t_mean / s_2) * (t_mean - t_mean ** 2 - s_2)
    b = ((1 - t_mean)/s_2) * (t_mean - t_mean ** 2 - s_2)

    '''Histogram of t with beta-distribution'''
    # lnspc = np.linspace(min(t), max(t), m)
    # beta_dist = _beta_distribution(alpha=a, beta=b)
    # beta_values = list(map(beta_dist, lnspc))
    # fig, ax = plt.subplots()
    # ax.hist(t, bins=k, normed=True)
    # plt.plot(lnspc, beta_values)
    # plt.show()
    # fig.savefig('hist-t-beta.png')

    '''Histogram of t with library beta-distribution'''
    # fig, ax = plt.subplots()
    # ax.hist(f_n, bins=k, normed=True)
    # lnspc = np.linspace(f_n_min, f_n_max, m)
    # ab, bb, cb, db = stats.beta.fit(f_n)
    # pdf_beta = stats.beta.pdf(lnspc, ab, bb, cb, db)
    # plt.plot(lnspc, pdf_beta, label="Beta")
    # plt.show()
    # fig.savefig('hist-t-beta-lib.png')


def main():
    #empirical_analysis()
    checking_distribution()

    '''Checking of random function'''
    # count = [0] * 100
    # for i in range(1000000):
    #     val = random.randint(0, 99)
    #     count[val] += 1
    # p_val = np.array(count) / 1000000
    # x = np.arange(0, 100)
    # fig, ax = plt.subplots()
    # ax.scatter(x, p_val, marker='o', s=10, c="red", edgecolor='b')
    # plt.show()


if __name__ == "__main__":
    main()
