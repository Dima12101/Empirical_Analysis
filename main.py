import sys
import random

MAX_DISTANCE = sys.maxsize


def print_result(start, distance, n):
    print("Стоимость пути из начальной вершины до остальных:")
    for i in range(0, n):
        if distance[i] != sys.maxsize:
            print(f"{start} > {i} = {distance[i]}")
        else:
            print(f"{start} > {i} = маршрут недоступен")

def generate_graph(count_V, count_E):
    graph = list([0] * count_V for i in range(0, count_V))
    temp_count_E = 0
    for i in range(0, count_V):
        if temp_count_E != count_E:
            for j in range(0, count_V):
                if temp_count_E != count_E:
                    is_exist_E = random.randint(False, True)
                    graph[i][j] = random.randint(1, 100) * is_exist_E
                    temp_count_E += is_exist_E
                else:
                    break
        else:
            break
    return graph


def Dijkstra(Graph, n, start):
    # Базовая операция: сравнение

    # Инициализация алгоритма
    visited = [False] * n
    distance = [MAX_DISTANCE] * n
    distance[start] = 0

    min_distance = 0
    index_node = start

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
    # Graph = generate_graph(10, 30)
    # for row in Graph:
    #     print(row)
    n = 6
    start = 0
    Graph = [
		[0, 1, 4, 0, 2, 0],
		[0, 0, 0, 9, 0, 0],
		[4, 0, 0, 7, 0, 0],
		[0, 9, 7, 0, 0, 2],
		[0, 0, 0, 0, 0, 8],
		[0, 0, 0, 0, 0, 0]
    ]
    distance = Dijkstra(Graph=Graph, n=n, start=start)
    print_result(start=start, distance=distance, n=n)

if __name__ == "__main__":
    main()
