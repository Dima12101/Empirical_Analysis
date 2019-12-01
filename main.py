import sys
import random
import matplotlib.pyplot as plt
import numpy as np



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
        # Базовая операция: сравнение

        counter = 0
        # Инициализация алгоритма
        n = len(Graph)

        visited = [False] * n
        distance = [self.MAX_DISTANCE] * n
        distance[start_node] = 0

        min_distance = 0
        index_node = start_node

        # Алгоритм
        while min_distance < self.MAX_DISTANCE:
            counter += 1  # <-- счётчик
            visited[index_node] = True
            # Обновление меток соседних вершин
            for i in range(0, n):
                new_distance = distance[index_node] + Graph[index_node][i]
                counter += 3    # <-- счётчик
                if not visited[i] and Graph[index_node][i] and new_distance < distance[i]:
                    distance[i] = new_distance
                
            # Поиск не посещённой вершины с минимальной меткой
            min_distance = self.MAX_DISTANCE
            for i in range(0, n):
                counter += 2  # <-- счётчик
                if not visited[i] and distance[i] < min_distance:
                    min_distance, index_node = distance[i], i
        counter += 1  # <-- счётчик
        return distance, counter


def print_result(distance, start_node):
    print("Стоимость пути из начальной вершины до остальных:")
    for i in range(0, len(distance)):
        if distance[i] != sys.maxsize:
            print(f"{start_node} > {i} = {distance[i]}")
        else:
            print(f"{start_node} > {i} = маршрут недоступен")


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
    fig.savefig(f'empirical_analysis.png')


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
    fig.savefig(f'empirical_f.png')


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
            _, f_on_n[j] = Dijkstra.algorithm(Graph, start_node)
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
        if (C1 > 0 and C2 > 0) and (C1 * g[i] >= n[i] and C2 * g[i] <= n[i] ** 3):
            print('n0:', n[i])
            print('C1:', C1)
            print('C2:', C2)
            _show_empirical_f_with_asymptotic(f, C1, C2)
            return True
    return False


def main():
    empirical_analysis()


if __name__ == "__main__":
    main()
