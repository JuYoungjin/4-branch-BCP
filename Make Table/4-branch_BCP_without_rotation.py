import numpy as np
from multiprocessing import Pool

"""
del0    del1    del2    del3
x0      x1      x2      x3      
|       |       |       |
+--k0---|       |       |
|       |       |       |
|       +--k1---|       |
|       |       |       |
|       |       +--k2---|
|       |       |       |
p0      p1      p2      p3
nav0    nav1    nav2    nav3
"""
# Calculate the 4-branch BCP by counting all cases
def count_all_value_4branch(del0, del1, del2, del3, nav0, nav1, nav2, nav3, bit=2):
    cnt = 0
    num = pow(2, bit)

    for x0 in range(num):
        for x1 in range(num):
            for x2 in range(num):
                for x3 in range(num):
                    for key0 in range(num):
                        for key1 in range(num):
                            for key2 in range(num):
                                x0_ = x0 ^ del0
                                x1_ = x1 ^ del1
                                x2_ = x2 ^ del2
                                x3_ = x3 ^ del3

                                p0 = (x0 + (x1 ^ key0)) % num
                                p1 = (x1 + (x2 ^ key1)) % num
                                p2 = (x2 + (x3 ^ key2)) % num
                                p3 = x3

                                p0_ = (x0_ + (x1_ ^ key0)) % num
                                p1_ = (x1_ + (x2_ ^ key1)) % num
                                p2_ = (x2_ + (x3_ ^ key2)) % num
                                p3_ = x3_

                                pp0 = p0 ^ nav0
                                pp1 = p1 ^ nav1
                                pp2 = p2 ^ nav2
                                pp3 = p3 ^ nav3

                                pp0_ = p0_ ^ nav0
                                pp1_ = p1_ ^ nav1
                                pp2_ = p2_ ^ nav2
                                pp3_ = p3_ ^ nav3

                                xx3 = pp3
                                xx2 = (pp2 - (xx3 ^ key2)) % num
                                xx1 = (pp1 - (xx2 ^ key1)) % num
                                xx0 = (pp0 - (xx1 ^ key0)) % num

                                xx3_ = pp3_
                                xx2_ = (pp2_ - (xx3_ ^ key2)) % num
                                xx1_ = (pp1_ - (xx2_ ^ key1)) % num
                                xx0_ = (pp0_ - (xx1_ ^ key0)) % num

                                if xx3_ ^ xx3 == del3:
                                    if xx2_ ^ xx2 == del2:
                                        if xx1_ ^ xx1 == del1:
                                            if xx0_ ^ xx0 == del0:
                                                cnt += 1

    sol = cnt / (num * num * num * num * num * num * num)
    return sol

# Make the 4-branch BCP Table
def four_bcp_matrix_gen(del0, del1, del2, del3, nav0, nav1, nav2, nav3):
    m = np.array(np.zeros((4096, 4096)))
    for carry in range(4096):
        c0 = (carry & 2048) >> 11
        c0_ = (carry & 1024) >> 10
        b0 = (carry & 512) >> 9
        b0_ = (carry & 256) >> 8

        c1 = (carry & 128) >> 7
        c1_ = (carry & 64) >> 6
        b1 = (carry & 32) >> 5
        b1_ = (carry & 16) >> 4

        c2 = (carry & 8) >> 3
        c2_ = (carry & 4) >> 2
        b2 = (carry & 2) >> 1
        b2_ = (carry & 1)

        if c0 ^ c0_ ^ b0 ^ b0_ != 0:
            continue
        if c1 ^ c1_ ^ b1 ^ b1_ != 0:
            continue
        if c2 ^ c2_ ^ b2 ^ b2_ != 0:
            continue

        for x0 in range(2):
            for x1 in range(2):
                for x2 in range(2):
                    for x3 in range(2):
                        for k0 in range(2):
                            for k1 in range(2):
                                for k2 in range(2):
                                    c0_next = 0
                                    c0_next_ = 0
                                    b0_next = 0
                                    b0_next_ = 0

                                    c1_next = 0
                                    c1_next_ = 0
                                    b1_next = 0
                                    b1_next_ = 0

                                    c2_next = 0
                                    c2_next_ = 0
                                    b2_next = 0
                                    b2_next_ = 0

                                    if x2 + (x3 ^ k2) + c2 >= 2:
                                        c2_next = 1
                                    if (x2 ^ del2) + (x3 ^ del3 ^ k2) + c2_ >= 2:
                                        c2_next_ = 1
                                    if ((x2 ^ (x3 ^ k2) ^ c2) ^ nav2) - (x3 ^ nav3 ^ k2) - b2 <= -1:
                                        b2_next = 1
                                    if ((x2 ^ del2) ^ (x3 ^ del3 ^ k2) ^ c2_ ^ nav2) - (
                                            x3 ^ del3 ^ nav3 ^ k2) - b2_ <= -1:
                                        b2_next_ = 1

                                    if x1 + (x2 ^ k1) + c1 >= 2:
                                        c1_next = 1
                                    if (x1 ^ del1) + (x2 ^ del2 ^ k1) + c1_ >= 2:
                                        c1_next_ = 1
                                    if (x1 ^ x2 ^ k1 ^ c1 ^ nav1) - (x2 ^ c2 ^ b2 ^ nav2 ^ nav3 ^ k1) - b1 <= -1:
                                        b1_next = 1
                                    if (x1 ^ del1 ^ x2 ^ del2 ^ k1 ^ c1_ ^ nav1) - (
                                            x2 ^ c2_ ^ b2_ ^ del2 ^ nav2 ^ nav3 ^ k1) - b1_ <= -1:
                                        b1_next_ = 1

                                    if x0 + (x1 ^ k0) + c0 >= 2:
                                        c0_next = 1
                                    if (x0 ^ del0) + (x1 ^ del1 ^ k0) + c0_ >= 2:
                                        c0_next_ = 1
                                    if (x0 ^ x1 ^ k0 ^ c0 ^ nav0) - (
                                            x1 ^ c1 ^ c2 ^ b1 ^ b2 ^ nav1 ^ nav2 ^ nav3 ^ k0) - b0 <= -1:
                                        b0_next = 1
                                    if (x0 ^ del0 ^ x1 ^ del1 ^ k0 ^ c0_ ^ nav0) - (
                                            x1 ^ c1_ ^ c2_ ^ b1_ ^ b2_ ^ del1 ^ nav1 ^ nav2 ^ nav3 ^ k0) - b0_ <= -1:
                                        b0_next_ = 1

                                    next_carry = (c0_next << 11) | (c0_next_ << 10) | (b0_next << 9) | (b0_next_ << 8) | (
                                            c1_next << 7) | (c1_next_ << 6) | (b1_next << 5) | (b1_next_ << 4) | (
                                            c2_next << 3) | (c2_next_ << 2) | (b2_next << 1) | (b2_next_ << 0)

                                    m[next_carry][carry] += 1

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
    m = m[append_list, :]
    m = m[:, append_list]
    m = m
    return m

# Save the 4-branch BCP Table
def four_bcp_matrix_save():
    B = []
    for i in range(256):
        del0 = (i & 128) >> 7
        del1 = (i & 64) >> 6
        del2 = (i & 32) >> 5
        del3 = (i & 16) >> 4

        nav0 = (i & 8) >> 3
        nav1 = (i & 4) >> 2
        nav2 = (i & 2) >> 1
        nav3 = (i & 1) >> 0

        m = four_bcp_matrix_gen(del0, del1, del2, del3, nav0, nav1, nav2, nav3)
        m = m
        B.append(m)
    B = np.array(B)
    np.save('four_bcp_matrix.npy', B)
    return B

# Calculating 4-branch BCP
def four_bcp_calculate(del0, del1, del2, del3, nav0, nav1, nav2, nav3, bit_num):
    B = np.load('four_bcp_matrix.npy')
    B = B / 128
    L = np.ones(512)
    C = np.zeros((512, 1))
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

def four_bcp_test(bit=3):
    num = pow(2, bit)
    cnt1 = 0
    cnt2 = 0
    for a1 in range(num):
        for a2 in range(num):
            for a3 in range(num):
                for a4 in range(num):
                    for b1 in range(num):
                        for b2 in range(num):
                            for b3 in range(num):
                                for b4 in range(num):
                                    p1 = count_all_value_4branch(a1, a2, a3, a4, b1, b2, b3, b4, bit=bit)
                                    p2 = four_bcp_calculate(a1, a2, a3, a4, b1, b2, b3, b4, bit_num=bit)
                                    if p1 == p2:
                                        cnt1 += 1
                                        # print(a1, a2, a3, b1, b2, b3)
                                        continue
                                    else:
                                        cnt2 += 1
                                        print(a1, a2, a3, a4, b1, b2, b3, b4)
                                        continue
    return cnt1, cnt2

"""
The following codes are compressing 4-branch BCP table by the methods in Kim et al. BCP paper
By this method, we can compress 4-branch BCP table triple times
"""
def four_bcp_matrix_gen_permuatate():
    B = np.load('four_bcp_matrix.npy')
    B_perm = []
    for t in range(256):
        a = B[t]
        b = np.zeros((512, 512))
        perm = [0, 7, 1, 6, 2, 5, 3, 4]
        for i in range(512):
            for j in range(512):
                quotient = i // 8
                remainder = i % 8
                point1 = (quotient * 8) + perm[remainder]

                quotient = j // 8
                remainder = j % 8
                point2 = (quotient * 8) + perm[remainder]

                b[i][j] = a[point1][point2]

        for i in range(256):
            for j in range(256):
                tmp1 = b[2 * i][2 * j]
                tmp2 = b[(2 * i) + 1][(2 * j) + 1]
                if tmp1 != tmp2:
                    print(i, j)
                    input("Something wrong")
                tmp3 = b[2 * i][(2 * j) + 1]
                tmp4 = b[(2 * i) + 1][2 * j]
                if tmp3 != tmp4:
                    print(i, j)
                    input("Something wrong")

        c = np.zeros((256, 256))
        for i in range(256):
            for j in range(256):
                c[i][j] = b[2 * i][2 * j] + b[2 * i][(2 * j) + 1]
        B_perm.append(c)
    B_perm = np.array(B_perm)
    np.save('four_bcp_matrix2.npy', B_perm)

def four_bcp_calculate2(del0, del1, del2, del3, nav0, nav1, nav2, nav3, bit_num):
    B = np.load('four_bcp_matrix2.npy')
    B = B / 128
    L = np.ones(256)
    C = np.zeros((256, 1))
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

def four_bcp_matrix_gen_permuatate2():
    B = np.load('four_bcp_matrix2.npy')
    B_perm = []
    for t in range(256):
        a = B[t]
        b = np.zeros((256, 256))
        perm = [0, 28, 1, 29, 2, 30, 3, 31, 4, 24, 5, 25, 6, 26, 7, 27, 8, 20, 9, 21 ,10, 22, 11, 23, 12, 16, 13, 17, 14, 18, 15, 19]
        for i in range(256):
            for j in range(256):
                quotient = i // 32
                remainder = i % 32
                point1 = (quotient * 32) + perm[remainder]

                quotient = j // 32
                remainder = j % 32
                point2 = (quotient * 32) + perm[remainder]

                b[i][j] = a[point1][point2]

        for i in range(128):
            for j in range(128):
                tmp1 = b[2 * i][2 * j]
                tmp2 = b[(2 * i) + 1][(2 * j) + 1]
                if tmp1 != tmp2:
                    print(i, j)
                    input("Something wrong")
                tmp3 = b[2 * i][(2 * j) + 1]
                tmp4 = b[(2 * i) + 1][2 * j]
                if tmp3 != tmp4:
                    print(i, j)
                    input("Something wrong")

        c = np.zeros((128, 128))
        for i in range(128):
            for j in range(128):
                c[i][j] = b[2 * i][2 * j] + b[2 * i][(2 * j) + 1]
        B_perm.append(c)
    B_perm = np.array(B_perm)
    np.save('four_bcp_matrix3.npy', B_perm)

def four_bcp_calculate3(del0, del1, del2, del3, nav0, nav1, nav2, nav3, bit_num):
    B = np.load('four_bcp_matrix3.npy')
    B = B / 128
    L = np.ones(128)
    C = np.zeros((128, 1))
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

def four_bcp_matrix_gen_permuatate3():
    B = np.load('four_bcp_matrix3.npy')
    B_perm = []
    for t in range(256):
        a = B[t]
        b = np.zeros((128, 128))
        perm = [0, 112, 1, 113, 2, 114, 3, 115, 4, 116, 5, 117, 6, 118, 7, 119, 8, 120, 9, 121, 10, 122, 11, 123, 12, 124, 13, 125, 14, 126, 15, 127, 16, 96, 17, 97, 18, 98, 19, 99, 20, 100, 21, 101, 22, 102, 23, 103, 24, 104, 25, 105, 26, 106, 27, 107, 28, 108, 29, 109, 30, 110, 31, 111, 32, 80, 33, 81, 34, 82, 35, 83, 36, 84, 37, 85, 38, 86, 39, 87, 40, 88, 41, 89, 42, 90, 43, 91, 44, 92, 45, 93, 46, 94, 47, 95, 48, 64, 49, 65, 50, 66, 51, 67, 52, 68, 53, 69, 54, 70, 55, 71, 56, 72, 57, 73, 58, 74, 59, 75, 60, 76, 61, 77, 62, 78, 63, 79]
        for i in range(128):
            for j in range(128):
                b[i][j] = a[perm[i]][perm[j]]

        for i in range(64):
            for j in range(64):
                tmp1 = b[2 * i][2 * j]
                tmp2 = b[(2 * i) + 1][(2 * j) + 1]
                if tmp1 != tmp2:
                    print(i, j)
                    input("Something wrong")
                tmp3 = b[2 * i][(2 * j) + 1]
                tmp4 = b[(2 * i) + 1][2 * j]
                if tmp3 != tmp4:
                    print(i, j)
                    input("Something wrong")

        c = np.zeros((64, 64))
        for i in range(64):
            for j in range(64):
                c[i][j] = b[2 * i][2 * j] + b[2 * i][(2 * j) + 1]
        B_perm.append(c)
    B_perm = np.array(B_perm)
    np.save('four_bcp_matrix4.npy', B_perm)

def four_bcp_calculate4(del0, del1, del2, del3, nav0, nav1, nav2, nav3, bit_num):
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

















