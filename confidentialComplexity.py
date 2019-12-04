from scipy import stats, special, integrate
import matplotlib.pyplot as plt
import numpy as np
import math

from algorithmDijkstra import Algorithm_Dijkstra

'''------------checking_distribution---------------'''


def _beta_distribution(alpha, beta):
    beta_coef = special.gamma(alpha + beta) / (special.gamma(alpha) * special.gamma(beta))
    return lambda x: beta_coef * math.pow(x, alpha - 1) * math.pow(1 - x, beta - 1)


def _try_other_dest(t, k):
    m = len(t)
    folder = 'confidentialComplexity_data/checking_distribution/lib_dist/'

    fig, ax = plt.subplots()
    ax.hist(t, bins=k, density=True)
    lnspc = np.linspace(min(t), max(t), m)
    parameters = stats.beta.fit(t)
    print(parameters)
    pdf_beta = stats.beta.pdf(lnspc, *parameters)
    ax.plot(lnspc, pdf_beta, label="B(alpha,beta)")
    ax.set_title('Бета-распрделение')
    ax.set_xlabel("t")
    ax.set_ylabel("$s_i$")
    ax.legend()
    plt.show()
    fig.savefig(folder+'hist-t-beta-lib.png')

    fig, ax = plt.subplots()
    ax.hist(t, bins=k, normed=True)
    lnspc = np.linspace(min(t), max(t), m)
    parameters = stats.f.fit(t)
    pdf = stats.f.pdf(lnspc, *parameters)
    ax.plot(lnspc, pdf, label="F(n,m)")
    ax.set_title('Распределение Фишера')
    ax.set_xlabel("t")
    ax.set_ylabel("$s_i$")
    ax.legend()
    plt.show()
    fig.savefig(folder+'hist-t-f-lib.png')

    fig, ax = plt.subplots()
    ax.hist(t, bins=k, normed=True)
    lnspc = np.linspace(min(t), max(t), m)
    parameters = stats.norm.fit(t)
    pdf = stats.norm.pdf(lnspc, *parameters)
    ax.plot(lnspc, pdf, label="N(a,sigma)")
    ax.set_title('Нормальное распределение')
    ax.set_xlabel("t")
    ax.set_ylabel("$s_i$")
    ax.legend()
    plt.show()
    fig.savefig(folder+'hist-t-norm-lib.png')

    fig, ax = plt.subplots()
    ax.hist(t, bins=k, normed=True)
    lnspc = np.linspace(min(t), max(t), m)
    parameters = stats.chi2.fit(t)
    pdf = stats.chi2.pdf(lnspc, *parameters)
    ax.plot(lnspc, pdf, label="Chi2(k)")
    ax.set_title('Распределение $хи^2$')
    ax.set_xlabel("t")
    ax.set_ylabel("$s_i$")
    ax.legend()
    plt.show()
    fig.savefig(folder+'hist-t-chi2-lib.png')


def _checking_distribution():
    Dijkstra = Algorithm_Dijkstra()
    folder = 'confidentialComplexity_data/checking_distribution'

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
    k = math.floor(1 + math.log2(m))
    print(k)
    #k = math.floor(math.pow(m, 1/3))

    '''Up/down load data'''
    # with open(f'{folder}/f-values-{m}.txt', 'w') as f_file:
    #     for f in f_n:
    #         f_file.write(str(f) + '\n')
    with open(f'{folder}/f-values-{m}.txt', 'r') as f_file:
        for i, line in enumerate(f_file.readlines()):
            f_n[i] = float(line)


    '''Histogram of f''' # Относительных частот
    fig, ax = plt.subplots()
    weights = np.ones_like(f_n) / float(len(f_n))
    ax.hist(f_n, bins=k, weights=weights)
    ax.set_xlabel("Трудоёмкость (f)")
    ax.set_ylabel("$w_i$")
    plt.show()
    fig.savefig(f'{folder}/hist-f-{m}.png')

    '''Some parameters'''
    f_n_mean = sum(f_n) / m
    f_n_min = min(f_n)
    f_n_max = max(f_n)

    '''f --> t'''
    t = (np.array(f_n) - f_n_min) / (f_n_max - f_n_min)
    '''Histogram of t''' # Относительных частот
    fig, ax = plt.subplots()
    weights = np.ones_like(f_n) / float(len(f_n))
    ax.hist(t, bins=k, weights=weights)
    ax.set_xlabel("t")
    ax.set_ylabel("$w_i$")
    plt.show()
    fig.savefig(f'{folder}/hist-t-{m}.png')

    '''Counting of ALPHA and BETA for beta-distribution'''
    t_mean = (f_n_mean - f_n_min) / (f_n_max - f_n_min)
    s_2 = sum(((np.array(f_n) - f_n_mean) ** 2) / ((f_n_max - f_n_min) ** 2)) / (m - 1)
    print(t_mean, s_2)
    a = (t_mean / s_2) * (t_mean - t_mean ** 2 - s_2)
    b = ((1 - t_mean)/s_2) * (t_mean - t_mean ** 2 - s_2)
    print(a, b)

    '''Histogram distribution of t with beta-distribution'''
    lnspc = np.linspace(min(t), max(t), m)
    beta_dist = _beta_distribution(alpha=a, beta=b)
    beta_values = list(map(beta_dist, lnspc))
    fig, ax = plt.subplots()
    ax.hist(t, bins=k,  density=True)
    ax.plot(lnspc, beta_values, label='$b(t, alpha, beta)$')
    ax.set_xlabel("t")
    ax.set_ylabel("$s_i$")
    ax.legend()
    plt.show()
    fig.savefig(f'{folder}/hist-t-beta-{m}.png')

    '''Histogram of t with library distribution'''
    _try_other_dest(t, k)

    weights = np.ones_like(t) / float(len(t))
    w, bins, _ = plt.hist(t, bins=k, weights=weights)
    beta_dist = _beta_distribution(alpha=a, beta=b)
    p = np.array(list(integrate.quad(beta_dist, bins[i], bins[i + 1])[0] for i in range(len(bins) - 1)))

    fig, ax = plt.subplots()
    ax.scatter(range(1, k + 1), w, marker='o', s=10, c="green", edgecolor='b')
    ax.plot(range(1, k + 1), w, color="green", linestyle='--', label="$w_i$")
    ax.scatter(range(1, k + 1), p, marker='o', s=10, c="red", edgecolor='b')
    ax.plot(range(1, k + 1), p, color="red", linestyle='--', label="$p_i$")
    ax.set_xlabel("Номер сегмента")
    ax.set_ylabel("Частота")
    ax.legend()
    plt.show()
    fig.savefig(f'{folder}/p-beta_and_t-{m}.png')

    Hi_2_view = sum((((w - p) ** 2) / p)) * m
    print('Hi_2_view', Hi_2_view)

    # Число степеней свободы
    st = k - 1 - 2
    # Уровень значимости
    level = 0.05
    Hi_2_critical = stats.chi2.ppf(1 - level, st)
    print('Hi_2_critical', Hi_2_critical)


'''------------------------------------------------'''


'''-----------------For regression------------------'''


def _getting_data_t_mean_and_s_2(n_start, n_end, n_step):
    Dijkstra = Algorithm_Dijkstra()

    m = 1000
    repeats = 100

    number_n = ((n_end - n_start) // n_step) + 1

    t_mean = [0] * number_n
    t_s2 = [0] * number_n

    for num, n in enumerate(range(n_start, n_end + 1, n_step)):
        f_n = [0] * m
        for i in range(m):
            Graph, start_node = Dijkstra.generate_data(n)
            f_repeats = [0] * repeats
            for j in range(repeats):
                _, _, f_repeats[j] = Dijkstra.algorithm(Graph, start_node)
            f_n[i] = sum(f_repeats) / repeats
        f_n_mean = sum(f_n) / m
        f_n_min = min(f_n)
        f_n_max = max(f_n)
        t_mean[num] = (f_n_mean - f_n_min) / (f_n_max - f_n_min)
        t_s2[num] = sum(((np.array(f_n) - f_n_mean) ** 2) / ((f_n_max - f_n_min) ** 2)) / (m - 1)
    return t_mean, t_s2


def _getting_data_f_up_f_down(n_start, n_end, n_step):
    Dijkstra = Algorithm_Dijkstra()

    m = 1000
    repeats = 100

    number_n = ((n_end - n_start) // n_step) + 1

    f_up = [0] * number_n
    f_down = [0] * number_n

    for num, n in enumerate(range(n_start, n_end + 1, n_step)):
        f_n = [0] * m
        for i in range(m):
            Graph, start_node = Dijkstra.generate_data(n)
            f_repeats = [0] * repeats
            for j in range(repeats):
                _, _, f_repeats[j] = Dijkstra.algorithm(Graph, start_node)
            f_n[i] = sum(f_repeats) / repeats
        f_up[num] = min(f_n)
        f_down[num] = max(f_n)

    return f_up, f_down


'''------------------------------------------------'''


def confidential_complexity():

    _checking_distribution()

    '''Getting data of t_mean and s_2'''
    # t_mea'''Getting data of t_mean and s_2'''n_n, t_s2_n = _getting_data_t_mean_and_s_2(n_start=10, n_end=100, n_step=5)
    # with open('t_mean_s2_excel.txt', 'w') as data:
    #     for t_mean in t_mean_n:
    #         data.write(f"{str(t_mean).replace('.',',')}\t")
    #     data.write('\n')
    #     for t_s2 in t_s2_n:
    #         data.write(f"{str(t_s2).replace('.',',')}\t")


    t_mean = lambda n: 0.143 * math.log1p(n) - 0.1672
    s_2 = lambda n: 0.1129 * math.exp(-0.02 * n)

    '''Parameters of beta-distribution'''
    a = lambda n: (t_mean(n) / s_2(n)) * (t_mean(n) - t_mean(n) ** 2 - s_2(n))
    b = lambda n: ((1 - t_mean(n)) / s_2(n)) * (t_mean(n) - t_mean(n) ** 2 - s_2(n))

    range_n = range(10, 401)
    a_values = list(map(a, range_n))
    b_values = list(map(b, range_n))
    fig, ax = plt.subplots()
    ax.set_xlabel("n")
    ax.plot(range_n, a_values, color="orange", label="$a(n)$")
    ax.plot(range_n, b_values, color="blue", label="$b(n)$")
    ax.legend()
    plt.show()
    fig.savefig('confidentialComplexity_data/plot_a_b.png')

    '''Function x_Y'''
    # Уровень доверия
    Y = 0.95
    x_Y = lambda n: stats.beta.ppf(Y, a(n), b(n))
    range_n = range(10, 401)
    x_Y_values = list(map(x_Y, range_n))
    fig, ax = plt.subplots()
    ax.set_xlabel("n")
    ax.plot(range_n, x_Y_values, color="blue", label="$x_Y(n)$")
    ax.legend()
    plt.show()
    fig.savefig('confidentialComplexity_data/plot_x_Y.png')

    '''Getting data of f_up and f_down'''
    # f_up_n, f_down_n = _getting_data_f_up_f_down(n_start=10, n_end=100, n_step=5)
    # with open('f_up_f_down_excel.txt', 'w') as data:
    #     for f_up in f_up_n:
    #         data.write(f"{str(f_up).replace('.',',')}\t")
    #     data.write('\n')
    #     for f_down in f_down_n:
    #         data.write(f"{str(f_down).replace('.',',')}\t")

    f_up = lambda n: 0.0004 * n ** 2 + 0.004 * n + 0.1224
    f_down = lambda n: 9e-05 * n ** 2 - 0.0006 * n + 0.0025

    '''Function f_Y'''
    f_Y = lambda n: f_down(n) + x_Y(n) * (f_up(n) - f_down(n))
    range_n = range(10, 401)
    f_up_values = list(map(f_up, range_n))
    f_Y_values = list(map(f_Y, range_n))
    f_down_values = list(map(f_down, range_n))
    fig, ax = plt.subplots()
    ax.set_xlabel("n")
    ax.plot(range_n, f_up_values, linestyle='--', color="red", label="up $f(n)$")
    ax.plot(range_n, f_Y_values, color="blue", label="$f_Y(n)$")
    ax.plot(range_n, f_down_values, linestyle='--', color="green", label="down $f(n)$")
    ax.legend()
    plt.show()
    fig.savefig('confidentialComplexity_data/plot_f_Y.png')