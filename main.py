import sys
import random


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
    distance = [sys.maxsize] * n
    distance[start] = 0

    # Алгоритм (максимальное число итерация = n-1)
    for iter in range(0, n - 1):
        # Поиск не посещённой вершины с минимальной меткой (за 2n)
        min, index = sys.maxsize, None
        for i in range(0, n):
            if not visited[i] and distance[i] < min:
                min, index = distance[i], i
        visited[index] = True
        # +1
        if distance[index] != sys.maxsize:
            # за (2 + 1)n в худшем случае
            for i in range(0, n):
                if not visited[i] and Graph[index][i]:
                    new_distance = distance[index] + Graph[index][i]
                    if new_distance < distance[i]:
                        distance[i] = new_distance
        else:
            break
    """Тогда
    В лучшем случае (полностью не связный граф): 2n                         <-- линейный класс
    В худшем случае (полный граф): (n - 1)(2n + 3n + 1) = (n - 1)(5n + 1)   <-- квадратичный класс
    """
    return distance


def main():
    Graph = generate_graph(10, 30)
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
    # distance = Dijkstra(Graph=Graph, n=n, start=start)
    # print_result(start=start, distance=distance, n=n)

if __name__ == "__main__":
    main()
