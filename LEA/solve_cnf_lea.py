from pycryptosat import *
from LEA_make_cnf import *
import numpy as np
from math import log2


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

def solve_dc_lea(Nr, prob_val, multiple=0):
    lea = LEA(n=128, k=128)
    model_name = 'model_file\\dc\\LEA_dc_round' + str(Nr) + '_prob' + str(prob_val) + '.cnf'

    clause, var_num, prob_num = lea.diff_make_cnf(Nr=Nr, prob_val=prob_val, model_name=model_name)

    solver = Solver(threads=15)
    with open(model_name, 'r') as x:
        for line in x:
            line = line.strip()
            out = [int(x) for x in line.split()[:-1]]
            solver.add_clause(out)

    # Determine SAT or UNSAT
    if multiple == 0:
        solution = solver.solve()
        print(solution[0])
        return solution, lea
    # Find Different Input and Output
    else:
        inp0 = lea.load_diff_var(r=0, num=0)
        inp1 = lea.load_diff_var(r=0, num=1)
        inp2 = lea.load_diff_var(r=0, num=2)
        inp3 = lea.load_diff_var(r=0, num=3)
        inp = inp0 + inp1 + inp2 + inp3

        out0 = lea.load_diff_var(r=Nr, num=0)
        out1 = lea.load_diff_var(r=Nr, num=1)
        out2 = lea.load_diff_var(r=Nr, num=2)
        out3 = lea.load_diff_var(r=Nr - 1, num=0)
        out = out0 + out1 + out2 + out3
        in_and_out = inp + out

        solution = solver.msolve_selected(max_nr_of_solutions=10000, var_selected=[])
        return solution, lea

def solve_dc_add_trail(Nr, prob_val, fix_val, in_or_out):
    lea = LEA(n=128, k=128)
    model_name = 'model_file\\dc\\LEA_amp_dc_round' + str(Nr) + '_prob_under' + str(prob_val) + '.cnf'

    if in_or_out == 'inp':
        cluase, var_num, prob_num = lea.diff_make_cnf_inp_fix(Nr=Nr, prob_val=prob_val,
                                                                        model_name=model_name,
                                                                        inp_val=fix_val)
        inp0 = lea.load_diff_var(r=0, num=0)
        inp1 = lea.load_diff_var(r=0, num=1)
        inp2 = lea.load_diff_var(r=0, num=2)
        inp3 = lea.load_diff_var(r=0, num=3)
        inp = inp0 + inp1 + inp2 + inp3
        others = inp
    elif in_or_out == 'out':
        cluase, var_num, prob_num = lea.diff_make_cnf_out_fix(Nr=Nr, prob_val=prob_val,
                                                                        model_name=model_name,
                                                                        out_val=fix_val)
        out0 = lea.load_diff_var(r=Nr, num=0)
        out1 = lea.load_diff_var(r=Nr, num=1)
        out2 = lea.load_diff_var(r=Nr, num=2)
        out3 = lea.load_diff_var(r=Nr - 1, num=0)
        out = out0 + out1 + out2 + out3
        others = out
    else:
        print("ERROR")
        return -1, -1

    solver = Solver(threads=16)
    with open(model_name, 'r') as x:
        for line in x:
            line = line.strip()
            out = [int(x) for x in line.split()[:-1]]
            solver.add_clause(out)

    var = []

    for i in range(1, lea.cnt):
        var.append(i)
    var = set(var)
    others = set(others)
    var = var - others
    var = list(var)

    solution = solver.msolve_selected(max_nr_of_solutions=10000, var_selected=var)
    return solution, lea

def lea_dc_add_prob_cal(r1, r2, r1_idx, r2_idx, r1_prob, r2_prob, r1_max_prob, r2_max_prob):
    # r1 part
    lea_inp_list = np.load('solution_file\\BCP_value\\lea_inp_list_' + str(r1) + '_' + str(r1_prob) + '.npy')
    r1_inp = lea_inp_list[r1_idx]
    r1_sol, r1_model = solve_dc_add_trail(Nr=r1, prob_val=r1_max_prob, fix_val=r1_inp, in_or_out='inp')

    n = 128
    w = n // 4

    r1_out = []
    for sol in r1_sol:
        x0 = r1_model.load_diff_var(r=r1, num=0)
        x1 = r1_model.load_diff_var(r=r1, num=1)
        x2 = r1_model.load_diff_var(r=r1, num=2)
        x3 = r1_model.load_diff_var(r=(r1 - 1), num=0)
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
            prob_var += r1_model.load_prob_var(r, num=0)
            prob_var += r1_model.load_prob_var(r, num=1)
            prob_var += r1_model.load_prob_var(r, num=2)
        for var in prob_var:
            if var in sol:
                p += 1
            else:
                p += 0

        (a0, a1, a2, a3) = bit_to_int(tmp, w)
        out_val = (a0, a1, a2, a3, p)
        r1_out.append(out_val)

    # r2 part
    lea_out_list = np.load('solution_file\\BCP_value\\lea_out_list_' + str(r2) + '_' + str(r2_prob) + '.npy')
    r2_out = lea_out_list[r2_idx]
    r2_sol, r2_model = solve_dc_add_trail(Nr=r2, prob_val=r2_max_prob, fix_val=r2_out, in_or_out='out')

    r2_inp = []
    for sol in r2_sol:
        x0 = r2_model.load_diff_var(r=0, num=0)
        x1 = r2_model.load_diff_var(r=0, num=1)
        x2 = r2_model.load_diff_var(r=0, num=2)
        x3 = r2_model.load_diff_var(r=0, num=3)
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
            prob_var += r2_model.load_prob_var(r, num=0)
            prob_var += r2_model.load_prob_var(r, num=1)
            prob_var += r2_model.load_prob_var(r, num=2)
        for var in prob_var:
            if var in sol:
                p += 1
            else:
                p += 0

        (a0, a1, a2, a3) = bit_to_int(tmp, w)
        inp_val = (a0, a1, a2, a3, p)
        r2_inp.append(inp_val)

    print("E1 Trail num :", len(r1_out))
    print("E1 Trail num :", len(r2_inp))

    fpp = open("MAX_SAVE.txt", "w")
    sol = 0
    max = 0
    for i in range(len(r1_out)):
        (a0, a1, a2, a3, p1) = r1_out[i]
        print(i, end=' ')
        if i % 10 == 0:
            print()
        for j in range(len(r2_inp)):
            (b0, b1, b2, b3, p2) = r2_inp[j]

            aa0 = a3
            aa1 = a2
            aa2 = a1
            aa3 = a0

            bb0 = rot_left(b2, 3, w)
            bb1 = rot_left(b1, 5, w)
            bb2 = rot_right(b0, 9, w)
            bb3 = b3

            bcp = four_bcp_calculate_no_rot(del0=aa0, del1=aa1, del2=aa2, del3=aa3,
                                            nav0=bb0, nav1=bb1, nav2=bb2, nav3=bb3, bit_num=w)
            tmp = bcp * pow(2, -(2 * p1)) * pow(2, -(2 * p2))
            if tmp > max:
                max = tmp
                max_r1_inp = r1_inp
                max_r1_out = r1_out[i]
                max_r2_inp = r2_inp[j]
                max_r2_out = r2_out
                max_bcp = bcp

                print("max bcp : ", max_bcp, file=fpp)
                print("MAX E1 Inp : ", max_r1_inp, file=fpp)
                print("MAX E1 Out : ", max_r1_out, file=fpp)
                print("E1 prob : ", p1, file=fpp)
                print("MAX E2 Inp : ", max_r2_inp, file=fpp)
                print("MAX E2 Out : ", max_r2_out, file=fpp)
                print("E2 prob : ", p2, file=fpp)
                print("----------------------------\n", file=fpp)
            sol += tmp

    fpp.close()

    sol = log2(sol)
    return sol, r1_out, r2_inp

def solution_save(Nr, prob_val):
    solution, lea = solve_dc_lea(Nr=Nr, prob_val=prob_val, multiple=1)
    name_dict = lea.var_name_dict
    n = 128
    w = n // 4

    num = 0
    in_list = []
    out_list = []
    print('number of solution : ', len(solution))

    for sol in solution:
        solution_name = 'solution_file\\dc\\LEA_dc_r' + str(Nr) + '_prob' + str(prob_val) + '_num' + str(num) + '.txt'
        num += 1
        f = open(solution_name, 'w')
        for r in range(Nr + 1):
            x0 = lea.load_diff_var(r=r, num=0)
            x1 = lea.load_diff_var(r=r, num=1)
            x2 = lea.load_diff_var(r=r, num=2)
            if r == 0:
                x3 = lea.load_diff_var(r=r, num=3)
            else:
                x3 = lea.load_diff_var(r=r-1, num=0)
            x = x0 + x1 + x2 + x3

            # in_list, out_list append
            if r == 0:
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
            if r == Nr:
                out = [None] * n
                cnt = 0
                for var in x:
                    if var in sol:
                        out[cnt] = 1
                    else:
                        out[cnt] = 0
                    cnt += 1
                (a0, a1, a2, a3) = bit_to_int(out, w)
                out_val = (a0, a1, a2, a3)
                out_list.append(out_val)

            # make txt file
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

            if r == Nr:
                continue

            prob0 = lea.load_prob_var(r, 0)
            prob1 = lea.load_prob_var(r, 1)
            prob2 = lea.load_prob_var(r, 2)
            prob = prob0 + prob1 + prob2

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

    np.save('solution_file\\BCP_value\\lea_inp_list_' + str(Nr) + '_' + str(prob_val) + '.npy', in_list)
    np.save('solution_file\\BCP_value\\lea_out_list_' + str(Nr) + '_' + str(prob_val) + '.npy', out_list)

    return 0

def rot_left(x, num, bit):
    mask = pow(2, bit) - 1
    return ((x << num) | (x >> (bit - num))) & mask

def rot_right(x, num, bit):
    mask = pow(2, bit) - 1
    return ((x << (bit - num)) | (x >> num)) & mask

def boomerang_prob_optimize(r1, r2, r1_prob, r2_prob):
    n = 128
    w = n // 4

    bcp_inp_list = np.load('solution_file\\BCP_value\\lea_out_list_' + str(r1) + '_' + str(r1_prob) + '.npy')
    bcp_out_list = np.load('solution_file\\BCP_value\\lea_inp_list_' + str(r2) + '_' + str(r2_prob) + '.npy')

    bcp_max = 0
    state = ('bcp : 0', 0)
    for i in range(len(bcp_inp_list)):
        for j in range(len(bcp_out_list)):
            (a0, a1, a2, a3) = bcp_inp_list[i]
            (b0, b1, b2, b3) = bcp_out_list[j]

            aa0 = a3
            aa1 = a2
            aa2 = a1
            aa3 = a0

            bb0 = rot_left(b2, 3, w)
            bb1 = rot_left(b1, 5, w)
            bb2 = rot_right(b0, 9, w)
            bb3 = b3

            bcp_val = four_bcp_calculate_no_rot(del0=aa0, del1=aa1, del2=aa2, del3=aa3,
                                                nav0=bb0, nav1=bb1, nav2=bb2, nav3=bb3, bit_num=w)

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

def four_bcp_calculate_no_rot(del0, del1, del2, del3, nav0, nav1, nav2, nav3, bit_num):
    B = np.load('four_bcp_matrix_no_rot.npy')
    B = B / 128
    L = np.ones(64)
    C = np.zeros((64, 1))
    C[0] = 1

    mask = 0x1
    ans = C
    for i in range(bit_num - 1):
        t = (((del0 & mask) << 7) | ((del1 & mask) << 6) | ((del2 & mask) << 5) | ((del3 & mask) << 4) |
             ((nav0 & mask) << 3) | ((nav1 & mask) << 2) | ((nav2 & mask) << 1) | ((nav3 & mask) << 0))
        del0 >>= 1
        del1 >>= 1
        del2 >>= 1
        del3 >>= 1

        nav0 >>= 1
        nav1 >>= 1
        nav2 >>= 1
        nav3 >>= 1

        ans = B[t] @ ans
    ans = L @ ans
    return ans.sum()

def lea_last_round_diff(r2, r2_idx, r2_prob, r2_max_prob):
    lea_out_list = np.load('solution_file\\BCP_value\\lea_out_list_' + str(r2) + '_' + str(r2_prob) + '.npy')
    r2_out = lea_out_list[r2_idx]
    r2_sol, r2_model = solve_dc_add_trail(Nr=r2, prob_val=r2_max_prob, fix_val=r2_out, in_or_out='out')

    n = 128
    w = n // 4

    r2_inp = []
    for sol in r2_sol:
        x0 = r2_model.load_diff_var(r=r2 - 1, num=0)
        x1 = r2_model.load_diff_var(r=r2 - 1, num=1)
        x2 = r2_model.load_diff_var(r=r2 - 1, num=2)
        x3 = r2_model.load_diff_var(r=r2 - 1, num=3)
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
            prob_var += r2_model.load_prob_var(r, num=0)
            prob_var += r2_model.load_prob_var(r, num=1)
            prob_var += r2_model.load_prob_var(r, num=2)
        for var in prob_var:
            if var in sol:
                p += 1
            else:
                p += 0

        (a0, a1, a2, a3) = bit_to_int(tmp, w)
        inp_val = (a0, a1, a2, a3, p)
        r2_inp.append(inp_val)

    print("E2 out num :", len(r2_inp))
    r2_inp = set(r2_inp)
    r2_inp = list(r2_inp)
    print("E2 out num :", len(r2_inp))
    fpp = open("r2_out.txt", "w")
    for i in range(len(r2_inp)):
        print(r2_inp[i], file=fpp)
    fpp.close()










