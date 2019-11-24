import sys
import random
import time
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

    def algorithm_operations(self, Graph, start_node):
        # Базовая операция: сравнение

        counter = 0
        # Инициализация алгоритма
        n = len(Graph)

        visited = [False] * n
        distance = [self.MAX_DISTANCE] * n
        distance[start_node] = 0

        min_distance = 0
        index_node = start_node

        # Алгоритм (максимальное число итерация = n - 1; на каждой ит. +1 от условия while; и +1 для выхода)
        while min_distance < self.MAX_DISTANCE:
            counter += 1  # <-- счётчик
            visited[index_node] = True
            # Обновление меток соседних вершин (за 3n)
            for i in range(0, n):
                new_distance = distance[index_node] + Graph[index_node][i]
                counter += 3    # <-- счётчик
                if not visited[i] and Graph[index_node][i] and new_distance < distance[i]:
                    distance[i] = new_distance
                
            # Поиск не посещённой вершины с минимальной меткой (за 2n)
            min_distance = self.MAX_DISTANCE
            for i in range(0, n):
                counter += 2  # <-- счётчик
                if not visited[i] and distance[i] < min_distance:
                    min_distance, index_node = distance[i], i
        counter += 1  # <-- счётчик

        """Сложность
        В лучшем случае (граф не связанный со стартовой вершиной): (3n + 2n + 1) * 1 + 1 = 5n + 2 = Омега(n) <-- линейный класс
        В худшем случае (полный граф): (3n + 2n + 1) * (n - 1) + 1 = 5n^2 - 4n = O(n^2)                      <-- квадратичный класс
        """
        return distance, counter

    def algorithm_time(self, Graph, start_node):
        # Инициализация алгоритма
        n = len(Graph)

        visited = [False] * n
        distance = [self.MAX_DISTANCE] * n
        distance[start_node] = 0

        min_distance = 0
        index_node = start_node

        # Алгоритм
        time_start = time.time()
        while min_distance < self.MAX_DISTANCE:
            visited[index_node] = True
            for i in range(0, n):
                new_distance = distance[index_node] + Graph[index_node][i]
                if not visited[i] and Graph[index_node][i] and new_distance < distance[i]:
                    distance[i] = new_distance
            min_distance = self.MAX_DISTANCE
            for i in range(0, n):
                if not visited[i] and distance[i] < min_distance:
                    min_distance, index_node = distance[i], i
        time_end = time.time()

        return distance, time_end - time_start


def print_result(distance, start_node):
    print("Стоимость пути из начальной вершины до остальных:")
    for i in range(0, len(distance)):
        if distance[i] != sys.maxsize:
            print(f"{start_node} > {i} = {distance[i]}")
        else:
            print(f"{start_node} > {i} = маршрут недоступен")


RANGE_n = (10, 100)

def _show_empirical_f(empirical_f, C, name_file='empirical_analysis', title=''):
    # Show Omega(n), O(n^2), empirical_f
    x = np.arange(*RANGE_n)
    upperAsymptotic = C * x ** 2
    fig, ax = plt.subplots()
    ax.scatter(x, empirical_f, marker='o', s=10, c="red", edgecolor='b', label="$f(n)$")
    ax.plot(x, upperAsymptotic, color="green", linestyle='--', label="$Cn^2$")
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("Трудоёмкость")
    ax.legend()

    plt.show()
    fig.savefig(f'{name_file}.png')


def empirical_analysis(type_complexity='operations'):
    Dijkstra = Algorithm_Dijkstra()
    range_n = RANGE_n
    m = 10

    # get empirical f
    empirical_f = [None] * (range_n[1] - range_n[0])
    if type_complexity == 'operations':
        for i, n in enumerate(range(*range_n)):
            f_on_n = [None] * m
            for j in range(m):
                Graph, start_node = Dijkstra.generate_data(n)
                _, f_on_n[j] = Dijkstra.algorithm_operations(Graph, start_node)
            empirical_f[i] = sum(f_on_n) / m
    elif type_complexity == 'time':
        for i, n in enumerate(range(*range_n)):
            f_on_n = [None] * m
            for j in range(m):
                Graph, start_node = Dijkstra.generate_data(n)
                _, f_on_n[j] = Dijkstra.algorithm_time(Graph, start_node)
            empirical_f[i] = sum(f_on_n) / m

    # Search constant C
    x = np.arange(*RANGE_n)
    ratio = np.array(empirical_f) / (x ** 2)
    C = ratio.mean()

    print(C)
    _show_empirical_f(empirical_f, C,
                      name_file=f'empirical_analysis_{type_complexity}',
                      title=f'The empirical analysis (by {type_complexity})')


def main():
    #empirical_analysis(type_complexity='operations')
    empirical_analysis(type_complexity='time')


if __name__ == "__main__":
    main()
