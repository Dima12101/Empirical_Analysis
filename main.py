import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'algorithmDijkstra'))
sys.path.append(os.path.join(BASE_DIR, 'empiricalAnalysis'))

from empiricalAnalysis import empirical_analysis
from confidentialComplexity import confidential_complexity


def main():
    empirical_analysis()
    confidential_complexity()

    '''Checking of random function'''
    # count = [0] * 100
    # for i in range(1000000):
    #     val = random.randint(0, 99)
    #     count[val] += 1
    # p_val = np.array(count) / 1000000
    # x = np.arange(0, 100)
    # fig, ax = plt.subplots()
    # ax.scatter(x, p_val, marker='o', s=10, c="red", edgecolor='b')
    # ax.set_title("Анализ равномерной генерации значений\n при помощи функции $random$")
    # ax.set_xlabel("Значения")
    # ax.set_ylabel("Частотная встречаемость")
    # plt.show()
    # fig.savefig('random-check.png')


if __name__ == "__main__":
    main()
