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
        counter = 0
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
                    #counter += 1
                    new_distance = distance[min_node] + Graph[min_node][node]
                    if distance[node] > new_distance:
                        #counter += 1
                        Q.remove((distance[node], node))
                        heapq.heappush(Q, (new_distance, node))

                        distance[node] = new_distance
                        parent[node] = min_node

        time_end = time.time()
        t = (time_end - time_start) * 1000
        return distance, parent, t


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
    # Show Omega(n), O(n^2), empirical_f
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

    # get empirical f
    f = [None] * (range_n[1] - range_n[0])
    for i, n in enumerate(range(*range_n)):
        f_on_n = [None] * m
        for j in range(m):
            Graph, start_node = Dijkstra.generate_data(n)
            _, _, f_on_n[j] = Dijkstra.algorithm(Graph, start_node)
        f[i] = sum(f_on_n) / m
    _show_empirical_f(f)

    # For showing of table
    with open('data.txt', 'w') as data:
        for n in range(*range_n):
            data.write(f"{n}\t")
        data.write('\n')
        for f_i in f:
            data.write(f"{str(f_i).replace('.',',')}\t")

    # Search constants: C1 and C2
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


def checking_distribution():
    Dijkstra = Algorithm_Dijkstra()
    n = 50
    m = 20000
    f_n = [0] * m
    # for i in range(m):
    #     Graph, start_node = Dijkstra.generate_data(n)
    #     temp_f = [0] * 100
    #     for j in range(100):
    #         _, _, temp_f[j] = Dijkstra.algorithm(Graph, start_node)
    #     f_n[i] = sum(temp_f) / 100


    #k = math.floor(1 + math.log2(m))
    #k = math.floor(math.sqrt(m))
    k = math.floor(math.pow(m, 1/3))
    print(k)

    # with open('hist.txt', 'w') as data:
    #     step = (f_n_max - f_n_min) / k
    #     l = f_n_min
    #     while l + step <= f_n_max:
    #         count = 0
    #         for f in f_n:
    #             if l <= f < l + step:
    #                 count += 1
    #         data.write(f"{str(count / m).replace('.', ',')}\t")
    #         l += step
    # with open('f-2.txt', 'w') as f_file:
    #     for f in f_n:
    #         f_file.write(str(f) + '\n')

    with open('f-2.txt', 'r') as f_file:
        for i, line in enumerate(f_file.readlines()):
            f_n[i] = float(line)

    f_n_mean = np.array(f_n).mean()
    f_n_min = min(f_n)
    f_n_max = max(f_n)
    print(f_n_min, f_n_mean, f_n_max)

    # fig, ax = plt.subplots()
    # w, bins, _ = ax.hist(f_n, bins=k, normed=True)
    # print(w)
    # print(bins)
    # plt.show()
    # fig.savefig('hist-2.png')

    t = (np.array(f_n) - f_n_min) / (f_n_max - f_n_min)
    fig, ax = plt.subplots()
    ax.hist(t, bins=k, normed=True)
    plt.show()
    fig.savefig('hist-t.png')

    # t_mean = (f_n_mean - f_n_min) / (f_n_max - f_n_min)
    # s_2 = sum(((np.array(f_n) - f_n_mean) ** 2) / ((f_n_max - f_n_min) ** 2)) / (m - 1)
    #
    # a = (t_mean / s_2) * (t_mean - t_mean ** 2 - s_2)
    # b = ((1 - t_mean)/s_2) * (t_mean - t_mean ** 2 - s_2)

    # fig, ax = plt.subplots()
    # ax.hist(f_n, bins=k, normed=True)
    #
    # lnspc = np.linspace(f_n_min, f_n_max, m)
    # ab, bb, cb, db = stats.beta.fit(f_n)
    # pdf_beta = stats.beta.pdf(lnspc, ab, bb, cb, db)
    # plt.plot(lnspc, pdf_beta, label="Beta")
    # plt.show()
    # fig.savefig('hist-test-1.png')


def main():
    #empirical_analysis()
    checking_distribution()

    #print(special.gamma(0.5))

    # # Checking of random function
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
