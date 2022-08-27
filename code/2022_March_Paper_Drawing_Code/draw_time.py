import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def draw_conn_figure(idx):
    xls_file = pd.ExcelFile('cortocal_v2_conn.xlsx')
    table1 = xls_file.parse('Result')

    time_800 = np.zeros((3, 8))
    time_1200 = np.zeros((3, 8))
    time_1600 = np.zeros((3, 8))

    index = ['VT_1', 'VT_2', 'VT_3', 'VT_4', 'VT_1+\nreorder', 'VT_2+\nreorder', 'VT_3+\nreorder', 'VT_4+\nreorder']

    count = 0
    for i in index:
        time_800[0][count] = table1[6:7][i][6]
        time_800[1][count] = table1[7:8][i][7]
        time_800[2][count] = table1[8:9][i][8]

        time_1200[0][count] = table1[18:19][i][18]
        time_1200[1][count] = table1[19:20][i][19]
        time_1200[2][count] = table1[20:21][i][20]

        time_1600[0][count] = table1[30:31][i][30]
        time_1600[1][count] = table1[31:32][i][31]
        time_1600[2][count] = table1[32:33][i][32]
        count += 1

    blank = np.zeros(2)

    bar_data = np.zeros((4, 3))
    for i in range(4):
        bar_data[i][0] = time_800[idx][0 + i]
        bar_data[i][1] = time_1200[idx][0 + i]
        bar_data[i][2] = time_1600[idx][0 + i]

    print(bar_data)

    n_groups = 3

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.2

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, bar_data[0], bar_width,
                    alpha=opacity, color='#000000',
                    error_kw=error_config,
                    label='VT1')
    rects2 = ax.bar(index + bar_width, bar_data[1], bar_width,
                    alpha=opacity, color='#C0C0C0',
                    error_kw=error_config,
                    label='VT2')
    rects3 = ax.bar(index + bar_width + bar_width, bar_data[2], bar_width,
                    alpha=opacity, color='#F4A460',
                    error_kw=error_config,
                    label='VT3')

    rects4 = ax.bar(index + bar_width + bar_width + bar_width, bar_data[3], bar_width,
                    alpha=opacity, color='#1E90FF',
                    error_kw=error_config,
                    label='VT4')

    ax.set_xticks(index + 3 * bar_width / 3)
    ax.set_xticklabels(('800', '1200', '1600'))
    ax.legend()
    plt.yscale('symlog', base=2)
    plt.ylim((0, 1200))
    plt.xlabel(u"dim", fontsize=13)
    plt.ylabel(u'connection', fontsize=13)

    fig.tight_layout()
    if idx == 0:
        str1 = "max_count"
    else:
        str1 = "average_count"

    plt.savefig(str1 + '_result.png', dpi=1000)
    # plt.rcParams['savefig.dpi'] = 1000
    plt.show()


def draw_time_figure():
    import pandas as pd

    xls_file = pd.ExcelFile('cortical_v2_time.xlsx')
    table1 = xls_file.parse('Result')

    copy_before_send_800 = ()
    copy_after_recving_800 = ()
    compute_duration_800 = ()
    import numpy as np

    time_800 = np.zeros((5, 8))
    time_1200 = np.zeros((5, 8))
    time_1600 = np.zeros((5, 8))

    index = ['VT_1', 'VT_2', 'VT_3', 'VT_4', 'VT_1+\nreorder', 'VT_2+\nreorder', 'VT_3+\nreorder', 'VT_4+\nreorder']

    count = 0
    for i in index:
        time_800[0][count] = table1[0:1][i][0]
        time_800[1][count] = table1[1:2][i][1]
        time_800[2][count] = table1[2:3][i][2]
        time_800[3][count] = table1[3:4][i][3]
        time_800[4][count] = table1[4:5][i][4]

        time_1200[0][count] = table1[8:9][i][8]
        time_1200[1][count] = table1[0:10][i][9]
        time_1200[2][count] = table1[10:11][i][10]
        time_1200[3][count] = table1[11:12][i][11]
        time_1200[4][count] = table1[12:13][i][12]

        time_1600[0][count] = table1[16:17][i][16]
        time_1600[1][count] = table1[17:18][i][17]
        time_1600[2][count] = table1[18:19][i][18]
        time_1600[3][count] = table1[19:20][i][19]
        time_1600[4][count] = table1[20:21][i][20]

        count += 1

    import numpy as np
    import matplotlib.pyplot as plt

    time_800[1][1:] = time_800[1][1:] + time_800[0][1:] - time_800[0][0]
    time_1200[1][1:] = time_1200[1][1:] + time_1200[0][1:] - time_1200[0][0]
    time_1600[1][1:] = time_1600[1][1:] + time_1600[0][1:] - time_1600[0][0]

    blank = np.zeros(1)

    # 中间插入的是空白的两行
    N = 3 * 8 + 2 * 1
    ind = np.arange(N)

    def concatenate_arr(time_800, blank, time_1600, time_1200):
        time_800_blank = np.append(time_800, blank)
        time_1600_blank = np.append(time_1200, blank)
        time_800_1600 = np.append(time_800_blank, time_1600_blank)
        return np.append(time_800_1600, time_1600)

    communication_cost = concatenate_arr(time_800[1], blank, time_1600[1], time_1200[1])

    print(communication_cost)

    copy_before_send_cost = concatenate_arr(time_800[0], blank, time_1600[0], time_1200[0])
    copy_after_send_cost = concatenate_arr(time_800[2], blank, time_1600[2], time_1200[2])
    compute_cost = concatenate_arr(time_800[3], blank, time_1600[3], time_1200[3])

    speed = concatenate_arr(time_800[4], blank, time_1600[4], time_1200[4])
    width = 0.5
    plt.figure(figsize=(30, 8))
    p1 = plt.bar(ind, communication_cost, width, color='#F4A460')  # , yerr=menStd)

    p2 = plt.plot(ind[0:4], speed[0:4], color='g', linestyle='-', marker='o')

    p2_1 = plt.plot(ind[4:8], speed[4:8], color='g', linestyle='-', marker='o')

    p2_2 = plt.plot(ind[9:13], speed[9:13], color='g', linestyle='-', marker='o')
    p2_3 = plt.plot(ind[13:17], speed[13:17], color='g', linestyle='-', marker='o')

    p2_4 = plt.plot(ind[18:22], speed[18:22], color='g', linestyle='-', marker='o')
    p2_5 = plt.plot(ind[22:26], speed[22:26], color='g', linestyle='-', marker='o')

    plt.ylabel("dim", fontsize=15, fontweight='bold')

    plt.xticks(ind, (
        'VT1', 'VT2', 'VT3', 'VT4', 'VT1', 'VT2', 'VT3', 'VT4', ' ', 'VT1', 'VT2', 'VT3', 'VT4', 'VT1', 'VT2', 'VT3',
        'VT4', ' ', 'VT1', 'VT2', 'VT3', 'VT4', 'VT1', 'VT2', 'VT3', 'VT4'))

    plt.legend((p1[0], p2[0]),
               ('Inter-process communication', 'Slow Down Ratio'), loc='upper left',
               fontsize=15)

    plt.savefig('comm_result.png', dpi=1000, transparent=True)
    plt.show()


# 绘制max count
draw_conn_figure(0)

# 绘制average count
draw_conn_figure(2)

draw_time_figure()

