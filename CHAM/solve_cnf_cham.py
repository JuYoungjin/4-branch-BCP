from pycryptosat import *
from CHAM_make_cnf import *
import numpy as np
from math import log2

# Bit vector -> Integer
def bit_to_int(inp, w):
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(w):
        a += pow(2, i) * inp[i]
        b += pow(2, i) * inp[i + w]
        c += pow(2, i) * inp[i + 2 * w]
        d += pow(2, i) * inp[i + 3 * w]
    return (a, b, c, d)

# Find Differential Trail for CHAM 
def solve_dc_cham(n, Nr, prob_val, start, multiple=0):
    cham = CHAM(n=n, k=128)
    model_name = 'model_file\\dc\\cham' + str(n) + \
                 '\\CHAM_' + str(n) + '_dc_start_' + str(start) + '_round' + str(Nr) + '_prob' + str(prob_val) + '.cnf'

    cluase, var_num, prob_num = cham.diff_make_cnf(start=start, Nr=Nr, prob_val=prob_val, model_name=model_name)

    solver = Solver(threads=16)
    with open(model_name, 'r') as x:
        for line in x:
            line = line.strip()
            out = [int(x) for x in line.split()[:-1]]
            solver.add_clause(out)

    # Determine SAT or UNSAT
    if multiple == 0:
        solution = solver.solve()
        print(solution[0])
        return solution, cham
    # Find Different Input and Output
    else:
        inp0 = cham.load_diff_var(r=start, num=0)
        inp1 = cham.load_diff_var(r=start, num=1)
        inp2 = cham.load_diff_var(r=start, num=2)
        inp3 = cham.load_diff_var(r=start, num=3)
        inp = inp0 + inp1 + inp2 + inp3
        out3 = cham.load_diff_var(r=(start + Nr), num=3)
        if Nr == 1:
            out2 = cham.load_diff_var(r=start, num=3)
            out1 = cham.load_diff_var(r=start, num=2)
            out0 = cham.load_diff_var(r=start, num=1)
        elif Nr == 2:
            out2 = cham.load_diff_var(r=(start + 1), num=3)
            out1 = cham.load_diff_var(r=start, num=3)
            out0 = cham.load_diff_var(r=start, num=2)
        else:
            out2 = cham.load_diff_var(r=(start + Nr - 1), num=3)
            out1 = cham.load_diff_var(r=(start + Nr - 2), num=3)
            out0 = cham.load_diff_var(r=(start + Nr - 3), num=3)
        out = out0 + out1 + out2 + out3
        in_and_out = inp + out

        solution = solver.msolve_selected(max_nr_of_solutions=10000, var_selected=[])
        return solution, cham

# Consider many diff trail
def solve_dc_add_trail(n, Nr, prob_val, start, fix_val, in_or_out):
    cham = CHAM(n=n, k=128)
    model_name = 'model_file\\dc\\cham' + str(n) + \
                 '\\CHAM_' + str(n) + '_amp_dc_start_' + str(start) + '_round' + str(Nr) + '_prob' + str(prob_val) + '.cnf'

    if in_or_out == 'inp':
        cluase, var_num, prob_num = cham.diff_make_cnf_inp_fix(start=start, Nr=Nr, prob_val=prob_val,
                                                                         model_name=model_name, inp_val=fix_val)

        inp0 = cham.load_diff_var(r=start, num=0)
        inp1 = cham.load_diff_var(r=start, num=1)
        inp2 = cham.load_diff_var(r=start, num=2)
        inp3 = cham.load_diff_var(r=start, num=3)
        inp = inp0 + inp1 + inp2 + inp3
        others = inp
    elif in_or_out == 'out':
        cluase, var_num, prob_num = cham.diff_make_cnf_out_fix(start=start, Nr=Nr, prob_val=prob_val,
                                                                         model_name=model_name, out_val=fix_val)

        out3 = cham.load_diff_var(r=(start + Nr), num=3)
        if Nr == 1:
            out2 = cham.load_diff_var(r=start, num=3)
            out1 = cham.load_diff_var(r=start, num=2)
            out0 = cham.load_diff_var(r=start, num=1)
        elif Nr == 2:
            out2 = cham.load_diff_var(r=(start + 1), num=3)
            out1 = cham.load_diff_var(r=start, num=3)
            out0 = cham.load_diff_var(r=start, num=2)
        else:
            out2 = cham.load_diff_var(r=(start + Nr - 1), num=3)
            out1 = cham.load_diff_var(r=(start + Nr - 2), num=3)
            out0 = cham.load_diff_var(r=(start + Nr - 3), num=3)
        out = out0 + out1 + out2 + out3
        others = out
    else:
        print("Error")
        return -1, -1

    solver = Solver(threads=16)
    with open(model_name, 'r') as x:
        for line in x:
            line = line.strip()
            out = [int(x) for x in line.split()[:-1]]
            solver.add_clause(out)

    var = []

    for i in range(1, cham.cnt):
        var.append(i)
    var = set(var)
    others = set(others)
    var = var - others
    var = list(var)

    solution = solver.msolve_selected(max_nr_of_solutions=10000, var_selected=var)
    return solution, cham

# Calculate the probability
def cham_dc_add_prob_cal(n, r1, r2, r1_idx, r2_idx, r1_prob, r2_prob, r1_max_prob, r2_max_prob):
    # r1 part
    cham_inp_list = np.load('solution_file\\BCP_value\\cham' + str(n) + '_even_inp_list_' + str(r1) + '.npy')
    r1_inp = cham_inp_list[r1_idx]
    r1_sol, r1_model = solve_dc_add_trail(n=n, Nr=r1, prob_val=r1_max_prob, start=0, fix_val=r1_inp, in_or_out='inp')

    w = n // 4

    r1_out = []
    for sol in r1_sol:
        x3 = r1_model.load_diff_var(r=r1, num=3)
        if r1 == 1:
            x2 = r1_model.load_diff_var(r=0, num=3)
            x1 = r1_model.load_diff_var(r=0, num=2)
            x0 = r1_model.load_diff_var(r=0, num=1)
        elif r1 == 2:
            x2 = r1_model.load_diff_var(r=1, num=3)
            x1 = r1_model.load_diff_var(r=0, num=3)
            x0 = r1_model.load_diff_var(r=0, num=2)
        else:
            x2 = r1_model.load_diff_var(r=(r1 - 1), num=3)
            x1 = r1_model.load_diff_var(r=(r1 - 2), num=3)
            x0 = r1_model.load_diff_var(r=(r1 - 3), num=3)
        x = x0 + x1 + x2 + x3

        tmp = [None] * n
        cnt = 0
        for var in x:
            if var in sol:
                tmp[cnt] = 1
            else:
                tmp[cnt] = 0
            cnt += 1

        p = 0
        prob_var = []
        for r in range(r1):
            prob_var += r1_model.load_prob_var(r)
        for var in prob_var:
            if var in sol:
                p += 1
            else:
                p += 0

        (a0, a1, a2, a3) = bit_to_int(tmp, w)
        out_val = (a0, a1, a2, a3, p)
        r1_out.append(out_val)

    # r2 part
    if r1 % 2 == 0:
        alpha = 1
        beta = 8
        start = 1
        cham_out_list = np.load('solution_file\\BCP_value\\cham' + str(n) + '_odd_out_list_' + str(r2) + '.npy')
        r2_out = cham_out_list[r2_idx]
        r2_sol, r2_model = solve_dc_add_trail(n=n, Nr=r2, prob_val=r2_max_prob, start=start, fix_val=r2_out, in_or_out='out')
    else:
        alpha = 8
        beta = 1
        start = 0
        cham_out_list = np.load('solution_file\\BCP_value\\cham' + str(n) + '_even_out_list_' + str(r2) + '.npy')
        r2_out = cham_out_list[r2_idx]
        r2_sol, r2_model = solve_dc_add_trail(n=n, Nr=r2, prob_val=r2_max_prob, start=start, fix_val=r2_out, in_or_out='out')

    r2_inp = []
    for sol in r2_sol:
        x0 = r2_model.load_diff_var(r=start, num=0)
        x1 = r2_model.load_diff_var(r=start, num=1)
        x2 = r2_model.load_diff_var(r=start, num=2)
        x3 = r2_model.load_diff_var(r=start, num=3)
        x = x0 + x1 + x2 + x3

        tmp = [None] * n
        cnt = 0
        for var in x:
            if var in sol:
                tmp[cnt] = 1
            else:
                tmp[cnt] = 0
            cnt += 1

        p = 0
        prob_var = []
        for r in range(r2):
            prob_var += r2_model.load_prob_var(r)
        for var in prob_var:
            if var in sol:
                p += 1
            else:
                p += 0

        (a0, a1, a2, a3) = bit_to_int(tmp, w)
        inp_val = (a0, a1, a2, a3, p)
        r2_inp.append(inp_val)

    print("E1 Trail num :", len(r1_out))
    print("E2 Trail num :", len(r2_inp))

    sol = 0
    for i in range(len(r1_out)):
        (a0, a1, a2, a3, p1) = r1_out[i]
        print(i, end=' ')
        for j in range(len(r2_inp)):
            (b0, b1, b2, b3, p2) = r2_inp[j]

            aa0 = a0
            aa1 = a1
            aa2 = a2
            aa3 = rot_right(a3, alpha, w)

            bb0 = rot_right(b0, beta, w)
            bb1 = rot_right(b1, alpha, w)
            bb2 = rot_right(b2, beta, w)
            bb3 = rot_left(b3, alpha, w)

            bcp = four_bcp_calculate(del0=aa0, del1=aa1, del2=aa2, del3=aa3,
                                     nav0=bb1, nav1=bb2, nav2=bb3, nav3=bb0, alpha=alpha, beta=beta, bit_num=w)

            tmp = bcp * pow(2, -(2 * p1)) * pow(2, -(2 * p2))
            sol += tmp
    print('---------------------------------')
    sol = log2(sol)
    return sol, r1_out, r2_inp

# Save the solution
def solution_save(n, Nr, prob_val, even_or_odd):
    if even_or_odd == 'even':
        start = 0
    elif even_or_odd == 'odd':
        start = 1
    else:
        assert (False), "Even Odd wrong"
    solution, cham = solve_dc_cham(n=n, Nr=Nr, prob_val=prob_val, start=start, multiple=1)
    name_dict = cham.var_name_dict
    w = n // 4

    num = 0
    in_list = []
    out_list = []
    print('number of solution :', len(solution))

    for sol in solution:
        if even_or_odd == 'even':
            solution_name = 'solution_file\\dc\\cham' + str(n) + '\\CHAM_' + str(n) + \
                            '_dc_even_r' + str(Nr) + '_prob' + str(prob_val) + '_num' + str(num) + '.txt'
        else:
            solution_name = 'solution_file\\dc\\cham' + str(n) + '\\CHAM_' + str(n) + \
                            '_dc_odd_r' + str(Nr) + '_prob' + str(prob_val) + '_num' + str(num) + '.txt'

        num += 1
        f = open(solution_name, 'w')
        for r in range(start, start + Nr + 1):
            if r == start:
                x0 = cham.load_diff_var(r=start, num=0)
                x1 = cham.load_diff_var(r=start, num=1)
                x2 = cham.load_diff_var(r=start, num=2)
                x3 = cham.load_diff_var(r=start, num=3)
                x = x0 + x1 + x2 + x3
            elif r == start + 1:
                x0 = cham.load_diff_var(r=start, num=1)
                x1 = cham.load_diff_var(r=start, num=2)
                x2 = cham.load_diff_var(r=start, num=3)
                x3 = cham.load_diff_var(r=(start + 1), num=3)
                x = x0 + x1 + x2 + x3
            elif r == start + 2:
                x0 = cham.load_diff_var(r=start, num=2)
                x1 = cham.load_diff_var(r=start, num=3)
                x2 = cham.load_diff_var(r=(start + 1), num=3)
                x3 = cham.load_diff_var(r=(start + 2), num=3)
                x = x0 + x1 + x2 + x3
            else:
                x0 = cham.load_diff_var(r=(r - 3), num=3)
                x1 = cham.load_diff_var(r=(r - 2), num=3)
                x2 = cham.load_diff_var(r=(r - 1), num=3)
                x3 = cham.load_diff_var(r=r, num=3)
                x = x0 + x1 + x2 + x3

            if r == start:
                inp = [None] * n
                cnt = 0
                for var in x:
                    if var in sol:
                        inp[cnt] = 1
                    else:
                        inp[cnt] = 0
                    cnt += 1
                (a0, a1, a2, a3) = bit_to_int(inp, w)
                inp_val = (a0, a1, a2, a3)
                in_list.append(inp_val)
            if r == (start + Nr):
                inp = [None] * n
                cnt = 0
                for var in x:
                    if var in sol:
                        inp[cnt] = 1
                    else:
                        inp[cnt] = 0
                    cnt += 1
                (a0, a1, a2, a3) = bit_to_int(inp, (n // 4))
                out_val = (a0, a1, a2, a3)
                out_list.append(out_val)

            prob = cham.load_prob_var(r)

            for var in x:
                name = name_dict[var]
                if var in sol:
                    value = 1
                else:
                    value = 0
                f.write(name)
                f.write(" : ")
                f.write(str(value))
                f.write("\n")

            if r == start + Nr:
                continue

            for var in prob:
                name = name_dict[var]
                if var in sol:
                    value = 1
                else:
                    value = 0
                f.write(name)
                f.write(' : ')
                f.write(str(value))
                f.write('\n')

        f.close()
    in_list = np.array(in_list)
    out_list = np.array(out_list)

    if even_or_odd == 'even':
        np.save('solution_file\\BCP_value\\cham' + str(n) + '_even_inp_list_' + str(Nr) + '.npy', in_list)
        np.save('solution_file\\BCP_value\\cham' + str(n) + '_even_out_list_' + str(Nr) + '.npy', out_list)
    else:
        np.save('solution_file\\BCP_value\\cham' + str(n) + '_odd_inp_list_' + str(Nr) + '.npy', in_list)
        np.save('solution_file\\BCP_value\\cham' + str(n) + '_odd_out_list_' + str(Nr) + '.npy', out_list)

    return 0

# Find best boomerang trail
def Boomerang_prob_optimize(r1, r2, n=64):
    w = n // 4

    bcp_inp_list = np.load('solution_file\\BCP_value\\cham' + str(n) + '_even_out_list_' + str(r1) + '.npy')

    if r1 % 2 == 0:
        alpha = 1
        beta = 8
        bcp_out_list = np.load('solution_file\\BCP_value\\cham' + str(n) + '_odd_inp_list_' + str(r2) + '.npy')
    else:
        alpha = 8
        beta = 1
        bcp_out_list = np.load('solution_file\\BCP_value\\cham' + str(n) + '_even_inp_list_' + str(r2) + '.npy')

    bcp_max = 0
    state = ("bcp : 0", 0)
    for i in range(len(bcp_inp_list)):
        for j in range(len(bcp_out_list)):
            (a0, a1, a2, a3) = bcp_inp_list[i]
            (b0, b1, b2, b3) = bcp_out_list[j]

            a3 = rot_right(a3, alpha, w)
            b0 = rot_right(b0, beta, w)
            b1 = rot_right(b1, alpha, w)
            b2 = rot_right(b2, beta, w)
            b3 = rot_left(b3, alpha, w)

            bcp_val = four_bcp_calculate(del0=a0, del1=a1, del2=a2, del3=a3,
                                         nav0=b1, nav1=b2, nav2=b3, nav3=b0, alpha=alpha, beta=beta, bit_num=w)
            if bcp_val > bcp_max:
                bcp_max = bcp_val
                state = (i, j)

    print()
    print('----------------------------------------------------')
    print(r1, '+', r2, 'round Boomerang Attack')
    print("Maximum BCP : ", bcp_max)
    print(state[0], 'th r1 trail ---', state[1], 'th r2 trail')
    print('----------------------------------------------------')
    print()
    return bcp_max, state

# Calculating 4-branch BCP
def rot_left(x, num, bit):
    mask = pow(2, bit) - 1
    return ((x << num) | (x >> (bit - num))) & mask

def rot_right(x, num, bit):
    mask = pow(2, bit) - 1
    return ((x << (bit - num)) | (x >> num)) & mask

def vector_exapansion(vec):
    a = np.zeros((4096, 1))
    append_list = [0, 3, 5, 6, 9, 10, 12, 15, 48, 51, 53, 54, 57, 58, 60, 63, 80, 83, 85, 86, 89, 90, 92, 95, 96, 99,
                   101, 102, 105, 106, 108, 111, 144, 147, 149, 150, 153, 154, 156, 159, 160, 163, 165, 166, 169, 170,
                   172, 175, 192, 195, 197, 198, 201, 202, 204, 207, 240, 243, 245, 246, 249, 250, 252, 255, 768, 771,
                   773, 774, 777, 778, 780, 783, 816, 819, 821, 822, 825, 826, 828, 831, 848, 851, 853, 854, 857, 858,
                   860, 863, 864, 867, 869, 870, 873, 874, 876, 879, 912, 915, 917, 918, 921, 922, 924, 927, 928, 931,
                   933, 934, 937, 938, 940, 943, 960, 963, 965, 966, 969, 970, 972, 975, 1008, 1011, 1013, 1014, 1017,
                   1018, 1020, 1023, 1280, 1283, 1285, 1286, 1289, 1290, 1292, 1295, 1328, 1331, 1333, 1334, 1337, 1338,
                   1340, 1343, 1360, 1363, 1365, 1366, 1369, 1370, 1372, 1375, 1376, 1379, 1381, 1382, 1385, 1386, 1388,
                   1391, 1424, 1427, 1429, 1430, 1433, 1434, 1436, 1439, 1440, 1443, 1445, 1446, 1449, 1450, 1452, 1455,
                   1472, 1475, 1477, 1478, 1481, 1482, 1484, 1487, 1520, 1523, 1525, 1526, 1529, 1530, 1532, 1535, 1536,
                   1539, 1541, 1542, 1545, 1546, 1548, 1551, 1584, 1587, 1589, 1590, 1593, 1594, 1596, 1599, 1616, 1619,
                   1621, 1622, 1625, 1626, 1628, 1631, 1632, 1635, 1637, 1638, 1641, 1642, 1644, 1647, 1680, 1683, 1685,
                   1686, 1689, 1690, 1692, 1695, 1696, 1699, 1701, 1702, 1705, 1706, 1708, 1711, 1728, 1731, 1733, 1734,
                   1737, 1738, 1740, 1743, 1776, 1779, 1781, 1782, 1785, 1786, 1788, 1791, 2304, 2307, 2309, 2310, 2313,
                   2314, 2316, 2319, 2352, 2355, 2357, 2358, 2361, 2362, 2364, 2367, 2384, 2387, 2389, 2390, 2393, 2394,
                   2396, 2399, 2400, 2403, 2405, 2406, 2409, 2410, 2412, 2415, 2448, 2451, 2453, 2454, 2457, 2458, 2460,
                   2463, 2464, 2467, 2469, 2470, 2473, 2474, 2476, 2479, 2496, 2499, 2501, 2502, 2505, 2506, 2508, 2511,
                   2544, 2547, 2549, 2550, 2553, 2554, 2556, 2559, 2560, 2563, 2565, 2566, 2569, 2570, 2572, 2575, 2608,
                   2611, 2613, 2614, 2617, 2618, 2620, 2623, 2640, 2643, 2645, 2646, 2649, 2650, 2652, 2655, 2656, 2659,
                   2661, 2662, 2665, 2666, 2668, 2671, 2704, 2707, 2709, 2710, 2713, 2714, 2716, 2719, 2720, 2723, 2725,
                   2726, 2729, 2730, 2732, 2735, 2752, 2755, 2757, 2758, 2761, 2762, 2764, 2767, 2800, 2803, 2805, 2806,
                   2809, 2810, 2812, 2815, 3072, 3075, 3077, 3078, 3081, 3082, 3084, 3087, 3120, 3123, 3125, 3126, 3129,
                   3130, 3132, 3135, 3152, 3155, 3157, 3158, 3161, 3162, 3164, 3167, 3168, 3171, 3173, 3174, 3177, 3178,
                   3180, 3183, 3216, 3219, 3221, 3222, 3225, 3226, 3228, 3231, 3232, 3235, 3237, 3238, 3241, 3242, 3244,
                   3247, 3264, 3267, 3269, 3270, 3273, 3274, 3276, 3279, 3312, 3315, 3317, 3318, 3321, 3322, 3324, 3327,
                   3840, 3843, 3845, 3846, 3849, 3850, 3852, 3855, 3888, 3891, 3893, 3894, 3897, 3898, 3900, 3903, 3920,
                   3923, 3925, 3926, 3929, 3930, 3932, 3935, 3936, 3939, 3941, 3942, 3945, 3946, 3948, 3951, 3984, 3987,
                   3989, 3990, 3993, 3994, 3996, 3999, 4000, 4003, 4005, 4006, 4009, 4010, 4012, 4015, 4032, 4035, 4037,
                   4038, 4041, 4042, 4044, 4047, 4080, 4083, 4085, 4086, 4089, 4090, 4092, 4095]
    cnt = 0
    for i in range(4096):
        if i in append_list:
            a[i] = vec[cnt]
            cnt += 1
    return a

def vector_compression(vec):
    append_list = [0, 3, 5, 6, 9, 10, 12, 15, 48, 51, 53, 54, 57, 58, 60, 63, 80, 83, 85, 86, 89, 90, 92, 95, 96, 99,
                   101, 102, 105, 106, 108, 111, 144, 147, 149, 150, 153, 154, 156, 159, 160, 163, 165, 166, 169, 170,
                   172, 175, 192, 195, 197, 198, 201, 202, 204, 207, 240, 243, 245, 246, 249, 250, 252, 255, 768, 771,
                   773, 774, 777, 778, 780, 783, 816, 819, 821, 822, 825, 826, 828, 831, 848, 851, 853, 854, 857, 858,
                   860, 863, 864, 867, 869, 870, 873, 874, 876, 879, 912, 915, 917, 918, 921, 922, 924, 927, 928, 931,
                   933, 934, 937, 938, 940, 943, 960, 963, 965, 966, 969, 970, 972, 975, 1008, 1011, 1013, 1014, 1017,
                   1018, 1020, 1023, 1280, 1283, 1285, 1286, 1289, 1290, 1292, 1295, 1328, 1331, 1333, 1334, 1337, 1338,
                   1340, 1343, 1360, 1363, 1365, 1366, 1369, 1370, 1372, 1375, 1376, 1379, 1381, 1382, 1385, 1386, 1388,
                   1391, 1424, 1427, 1429, 1430, 1433, 1434, 1436, 1439, 1440, 1443, 1445, 1446, 1449, 1450, 1452, 1455,
                   1472, 1475, 1477, 1478, 1481, 1482, 1484, 1487, 1520, 1523, 1525, 1526, 1529, 1530, 1532, 1535, 1536,
                   1539, 1541, 1542, 1545, 1546, 1548, 1551, 1584, 1587, 1589, 1590, 1593, 1594, 1596, 1599, 1616, 1619,
                   1621, 1622, 1625, 1626, 1628, 1631, 1632, 1635, 1637, 1638, 1641, 1642, 1644, 1647, 1680, 1683, 1685,
                   1686, 1689, 1690, 1692, 1695, 1696, 1699, 1701, 1702, 1705, 1706, 1708, 1711, 1728, 1731, 1733, 1734,
                   1737, 1738, 1740, 1743, 1776, 1779, 1781, 1782, 1785, 1786, 1788, 1791, 2304, 2307, 2309, 2310, 2313,
                   2314, 2316, 2319, 2352, 2355, 2357, 2358, 2361, 2362, 2364, 2367, 2384, 2387, 2389, 2390, 2393, 2394,
                   2396, 2399, 2400, 2403, 2405, 2406, 2409, 2410, 2412, 2415, 2448, 2451, 2453, 2454, 2457, 2458, 2460,
                   2463, 2464, 2467, 2469, 2470, 2473, 2474, 2476, 2479, 2496, 2499, 2501, 2502, 2505, 2506, 2508, 2511,
                   2544, 2547, 2549, 2550, 2553, 2554, 2556, 2559, 2560, 2563, 2565, 2566, 2569, 2570, 2572, 2575, 2608,
                   2611, 2613, 2614, 2617, 2618, 2620, 2623, 2640, 2643, 2645, 2646, 2649, 2650, 2652, 2655, 2656, 2659,
                   2661, 2662, 2665, 2666, 2668, 2671, 2704, 2707, 2709, 2710, 2713, 2714, 2716, 2719, 2720, 2723, 2725,
                   2726, 2729, 2730, 2732, 2735, 2752, 2755, 2757, 2758, 2761, 2762, 2764, 2767, 2800, 2803, 2805, 2806,
                   2809, 2810, 2812, 2815, 3072, 3075, 3077, 3078, 3081, 3082, 3084, 3087, 3120, 3123, 3125, 3126, 3129,
                   3130, 3132, 3135, 3152, 3155, 3157, 3158, 3161, 3162, 3164, 3167, 3168, 3171, 3173, 3174, 3177, 3178,
                   3180, 3183, 3216, 3219, 3221, 3222, 3225, 3226, 3228, 3231, 3232, 3235, 3237, 3238, 3241, 3242, 3244,
                   3247, 3264, 3267, 3269, 3270, 3273, 3274, 3276, 3279, 3312, 3315, 3317, 3318, 3321, 3322, 3324, 3327,
                   3840, 3843, 3845, 3846, 3849, 3850, 3852, 3855, 3888, 3891, 3893, 3894, 3897, 3898, 3900, 3903, 3920,
                   3923, 3925, 3926, 3929, 3930, 3932, 3935, 3936, 3939, 3941, 3942, 3945, 3946, 3948, 3951, 3984, 3987,
                   3989, 3990, 3993, 3994, 3996, 3999, 4000, 4003, 4005, 4006, 4009, 4010, 4012, 4015, 4032, 4035, 4037,
                   4038, 4041, 4042, 4044, 4047, 4080, 4083, 4085, 4086, 4089, 4090, 4092, 4095]
    vec = vec[append_list, :]
    return vec

def four_bcp_calculate(del0, del1, del2, del3, nav0, nav1, nav2, nav3, alpha, beta, bit_num):
    B = np.load('four_bcp_calculate\\four_bcp_matrix.npy')
    B = B / 128
    R = np.load('four_bcp_calculate\\rotation_matrix.npy')
    L = np.load('four_bcp_calculate\\L_matrix.npy')
    C = np.load('four_bcp_calculate\\C_matrix.npy')

    sum = 0
    del3 = rot_left(rot_left(del3, beta, bit_num), alpha, bit_num)
    del2 = rot_left(rot_left(del2, beta, bit_num), alpha, bit_num)
    nav3 = rot_left(rot_left(nav3, beta, bit_num), alpha, bit_num)
    nav2 = rot_left(rot_left(nav2, beta, bit_num), alpha, bit_num)
    del1 = rot_left(del1, alpha, bit_num)
    nav1 = rot_left(nav1, alpha, bit_num)

    for j in range(64):
        del0_ = del0
        del1_ = del1
        del2_ = del2
        del3_ = del3
        nav0_ = nav0
        nav1_ = nav1
        nav2_ = nav2
        nav3_ = nav3

        Lt = L[j]
        Ct = C[j]

        mask = 0x1

        ans = Ct
        for i in range(bit_num - 1):
            t = (((del0_ & mask) << 7) | ((del1_ & mask) << 6) | ((del2_ & mask) << 5) | ((del3_ & mask) << 4) |
                 ((nav0_ & mask) << 3) | ((nav1_ & mask) << 2) | ((nav2_ & mask) << 1) | ((nav3_ & mask) << 0))
            del0_ >>= 1
            del1_ >>= 1
            del2_ >>= 1
            del3_ >>= 1

            nav0_ >>= 1
            nav1_ >>= 1
            nav2_ >>= 1
            nav3_ >>= 1

            if i == (alpha - 1):
                ans = vector_exapansion(ans)
                B_r0 = np.load('four_bcp_calculate\\four_bcp_last_matrix' + str(t) + '.npy')
                B_r0 = B_r0 / 128
                ans = B_r0 @ ans
                ans = R[0] @ ans
                ans = vector_compression(ans)
                continue
            if i == ((beta + alpha - 1) % bit_num):
                ans = vector_exapansion(ans)
                B_r1 = np.load('four_bcp_calculate\\four_bcp_last_matrix' + str(t) + '.npy')
                B_r1 = B_r1 / 128
                ans = B_r1 @ ans
                ans = R[1] @ ans
                ans = vector_compression(ans)
                continue

            ans = B[t] @ ans

        t = (((del0_ & mask) << 7) | ((del1_ & mask) << 6) | ((del2_ & mask) << 5) | ((del3_ & mask) << 4) |
             ((nav0_ & mask) << 3) | ((nav1_ & mask) << 2) | ((nav2_ & mask) << 1) | ((nav3_ & mask) << 0))
        last_B = np.load('four_bcp_calculate\\four_bcp_last_matrix' + str(t) +'.npy')
        last_B = last_B / 128
        ans = vector_exapansion(ans)
        ans = last_B @ ans

        ans = Lt @ ans
        ans = ans.sum()
        sum += ans
    return sum

