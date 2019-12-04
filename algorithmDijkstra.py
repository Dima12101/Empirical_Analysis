import sys
import time
import heapq
import random


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