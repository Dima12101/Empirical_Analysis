import sys
import random


RANGE_VALUES = (1, 10)

def _generate_graph(count_V, count_E):
    # Генерация не связанного графа
    graph = list([0] * count_V for i in range(0, count_V))
    # Добавление рёбер
    temp_count_E = 0
    while temp_count_E < count_E:
        # Случайным образом выбираем пару вершин
        begin_node = random.randint(0, count_V- 1)
        end_node = random.randint(0, count_V - 1)
        if graph[begin_node][end_node] == 0:
            # Случайным образом выясняем, есть ли между ними ребро
            is_exist_E = random.randint(False, True)
            if is_exist_E:
                # Создаём ребро
                graph[begin_node][end_node] = random.randint(*RANGE_VALUES)
                temp_count_E += 1
    return graph

def generate_data(n):
    # Случайным образом выбираем кол-во рёбер
    count_E = random.randint(0, n * (n + 1) / 2)
    # Случайным образом генерируем граф
    Graph = _generate_graph(count_V=n, count_E=count_E)
    # Случайным образом выбираем стартовую вершину
    start_node = random.randint(0, n - 1)

    return Graph, start_node

def print_result(start_node, distance, n):
    print("Стоимость пути из начальной вершины до остальных:")
    for i in range(0, n):
        if distance[i] != sys.maxsize:
            print(f"{start_node} > {i} = {distance[i]}")
        else:
            print(f"{start_node} > {i} = маршрут недоступен")

MAX_DISTANCE = sys.maxsize

def Dijkstra(Graph, n, start_node):
    # Базовая операция: сравнение

    # Инициализация алгоритма
    visited = [False] * n
    distance = [MAX_DISTANCE] * n
    distance[start_node] = 0

    min_distance = 0
    index_node = start_node

    # Алгоритм (максимальное число итерация = n - 1; на каждой ит. +1 от условия while; и +1 для выхода)
    while min_distance < MAX_DISTANCE:
        visited[index_node] = True
        # Обновление меток соседних вершин (за 3n)
        for i in range(0, n):
            new_distance = distance[index_node] + Graph[index_node][i]
            if not visited[i] and Graph[index_node][i] and new_distance < distance[i]:
                distance[i] = new_distance
            
        # Поиск не посещённой вершины с минимальной меткой (за 2n)
        min_distance = MAX_DISTANCE
        for i in range(0, n):
            if not visited[i] and distance[i] < min_distance:
                min_distance, index_node = distance[i], i

    """Тогда
    В лучшем случае (граф не связанный со стартовой вершиной): (3n + 2n + 1) * 1 + 1 = 5n + 2 = O(n)    <-- линейный класс
    В худшем случае (полный граф): (3n + 2n + 1) * (n - 1) + 1 = 5n^2 - 4n = O(n^2)                     <-- квадратичный класс
    """
    return distance


def main():
    n = 10
    Graph, start_node = generate_data(n)
    for row in Graph:
        print(row)
    # n = 6
    # start = 0
    # Graph = [
	# 	[0, 1, 4, 0, 2, 0],
	# 	[0, 0, 0, 9, 0, 0],
	# 	[4, 0, 0, 7, 0, 0],
	# 	[0, 9, 7, 0, 0, 2],
	# 	[0, 0, 0, 0, 0, 8],
	# 	[0, 0, 0, 0, 0, 0]
    # ]
    distance = Dijkstra(Graph=Graph, n=n, start_node=start_node)
    print_result(start_node=start_node, distance=distance, n=n)

if __name__ == "__main__":
    main()
