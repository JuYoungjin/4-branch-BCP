from make_tree import *

class LEA:
    def __init__(self, n=128, k=128):
        if k in [128, 196, 256]:
            self.n = n
            self.k = k
            self.w = n // 4
            self.cnt = 1
            self.prob_cnt = 0
            self.var_name_dict = dict()
        else:
            assert (False), "Wrong LEA version"

    def gen_diff_var(self, r, num):
        var_name = [None] * self.w
        var = [None] * self.w
        j = 0
        for i in range(num * self.w, (num + 1) * self.w):
            var_name[j] = 'diff_in_' + str(r) + '_' + str(i)
            var[j] = self.cnt
            self.var_name_dict[self.cnt] = var_name[j]
            self.cnt += 1
            j += 1
        return var

    def gen_prob_var(self, r, num):
        var_name = [None] * (self.w - 1)
        var = [None] * (self.w - 1)
        for i in range(self.w - 1):
            var_name[i] = 'diff_w' + str(num) + '_' + str(r) + '_' + str(i)
            var[i] = self.prob_cnt
            self.var_name_dict[self.prob_cnt] = var_name[i]
            self.prob_cnt += 1
        return var

    def load_diff_var(self, r, num):
        def search_key(dic, value):
            for key, val in dic.items():
                if val == value:
                    return key

        var_name = [None] * self.w
        var = [None] * self.w
        j = 0
        for i in range(num * self.w, (num + 1) * self.w):
            var_name[j] = 'diff_in_' + str(r) + '_' + str(i)
            var[j] = search_key(dic=self.var_name_dict, value=var_name[j])
            j += 1
        return var

    def load_prob_var(self, r, num):
        def search_key(dic, value):
            for key, val in dic.items():
                if val == value:
                    return key

        var_name = [None] * (self.w - 1)
        var = [None] * (self.w - 1)
        for i in range(self.w - 1):
            var_name[i] = 'diff_w' + str(num) + '_' + str(r) + '_' + str(i)
            var[i] = search_key(dic=self.var_name_dict, value=var_name[i])
        return var

    def modulo_addition(self, in1, in2, out, prob):
        clause = []

        # bit position 0
        clause += ["%d %d -%d 0" %(in1[0], in2[0], prob[0])]
        clause += ["-%d %d 0" %(in2[0], prob[0])]
        clause += ["-%d %d %d 0" %(in1[0], in2[0], out[0])]
        clause += ["%d %d -%d 0" %(in1[0], out[0], prob[0])]
        clause += ["-%d %d 0" %(out[0], prob[0])]
        clause += ["-%d -%d -%d 0" %(in1[0], in2[0], out[0])]

        for i in range(1, self.w - 1):
            clause += ["-%d -%d -%d %d %d %d 0" %(in1[i - 1], in2[i - 1], out[i - 1], in1[i], in2[i], out[i])]
            clause += ["%d %d %d -%d -%d -%d 0" %(in1[i - 1], in2[i - 1], out[i - 1], in1[i], in2[i], out[i])]
            clause += ["%d %d %d %d %d -%d 0" %(in1[i - 1], in2[i - 1], out[i - 1], in1[i], in2[i], prob[i])]
            clause += ["%d %d %d %d %d -%d 0" %(in1[i - 1], in2[i - 1], out[i - 1], in1[i], out[i], prob[i])]
            clause += ["%d %d %d %d %d -%d 0" %(in1[i - 1], in2[i - 1], out[i - 1], in2[i], out[i], prob[i])]
            clause += ["-%d -%d -%d -%d -%d -%d 0" %(in1[i - 1], in2[i - 1], out[i - 1], in1[i], in2[i], prob[i])]
            clause += ["-%d -%d -%d -%d -%d -%d 0" %(in1[i - 1], in2[i - 1], out[i - 1], in1[i], out[i], prob[i])]
            clause += ["-%d -%d -%d -%d -%d -%d 0" %(in1[i - 1], in2[i - 1], out[i - 1], in2[i], out[i], prob[i])]
            clause += ["-%d %d %d 0" %(in1[i], out[i], prob[i])]
            clause += ["%d -%d %d 0" %(in2[i], out[i], prob[i])]
            clause += ["%d -%d %d 0" %(in1[i], in2[i], prob[i])]
            clause += ["%d %d %d -%d 0" %(in1[i], in2[i], out[i], prob[i])]
            clause += ["-%d -%d -%d -%d 0" %(in1[i], in2[i], out[i], prob[i])]

        # bit position w-1
        clause += ['%d %d %d -%d %d %d 0' %(in1[self.w - 2], in2[self.w - 2], out[self.w - 2], in1[self.w - 1], in2[self.w - 1], out[self.w - 1])]
        clause += ['%d %d %d %d -%d %d 0' % (in1[self.w - 2], in2[self.w - 2], out[self.w - 2], in1[self.w - 1], in2[self.w - 1], out[self.w - 1])]
        clause += ['%d %d %d %d %d -%d 0' % (in1[self.w - 2], in2[self.w - 2], out[self.w - 2], in1[self.w - 1], in2[self.w - 1], out[self.w - 1])]
        clause += ['%d %d %d -%d -%d -%d 0' % (in1[self.w - 2], in2[self.w - 2], out[self.w - 2], in1[self.w - 1], in2[self.w - 1], out[self.w - 1])]
        clause += ['-%d -%d -%d %d %d %d 0' % (in1[self.w - 2], in2[self.w - 2], out[self.w - 2], in1[self.w - 1], in2[self.w - 1], out[self.w - 1])]
        clause += ['-%d -%d -%d %d -%d -%d 0' % (in1[self.w - 2], in2[self.w - 2], out[self.w - 2], in1[self.w - 1], in2[self.w - 1], out[self.w - 1])]
        clause += ['-%d -%d -%d -%d %d -%d 0' % (in1[self.w - 2], in2[self.w - 2], out[self.w - 2], in1[self.w - 1], in2[self.w - 1], out[self.w - 1])]
        clause += ['-%d -%d -%d -%d -%d %d 0' % (in1[self.w - 2], in2[self.w - 2], out[self.w - 2], in1[self.w - 1], in2[self.w - 1], out[self.w - 1])]

        return clause

    def xor(self, in1, in2, out):
        clause = []
        for i in range(self.w):
            clause += ['x%d %d %d 0' %(in1[i], in2[i], out[i])]
        return clause

    def rotation(self, inp, rot_num, word_size, left_or_right='left'):
        out = [None] * word_size
        # left rotation
        if left_or_right == 'left':
            for i in range(word_size):
                out[i] = inp[(i - rot_num) % word_size]
        elif left_or_right == 'right':
            for i in range(word_size):
                out[i] = inp[(i + rot_num) % word_size]
        return out

    def lea_diff_one_round(self, inp, r):
        clause = []

        prob0 = self.load_prob_var(r, 0)
        prob1 = self.load_prob_var(r, 1)
        prob2 = self.load_prob_var(r, 2)

        in0 = inp[0:self.w]
        in1 = inp[self.w:(2 * self.w)]
        in2 = inp[(2 * self.w):(3 * self.w)]
        in3 = inp[(3 * self.w):(4 * self.w)]

        out0 = self.load_diff_var(r+1, 0)
        out1 = self.load_diff_var(r+1, 1)
        out2 = self.load_diff_var(r+1, 2)
        out3 = in0
        out = out0 + out1 + out2 + out3

        tp1 = self.rotation(inp=out0, rot_num=9, word_size=self.w, left_or_right='right')
        tp2 = self.rotation(inp=out1, rot_num=5, word_size=self.w, left_or_right='left')
        tp3 = self.rotation(inp=out2, rot_num=3, word_size=self.w, left_or_right='left')

        clause += self.modulo_addition(in1=in0, in2=in1, out=tp1, prob=prob0)
        clause += self.modulo_addition(in1=in1, in2=in2, out=tp2, prob=prob1)
        clause += self.modulo_addition(in1=in2, in2=in3, out=tp3, prob=prob2)

        return clause, out

    def lea_diff_clause(self, end, amp=0):
        clause = []
        for r in range(end + 1):
            if r != 0:
                _ = self.gen_diff_var(r=r, num=0)
                _ = self.gen_diff_var(r=r, num=1)
                _ = self.gen_diff_var(r=r, num=2)
            else:
                _ = self.gen_diff_var(r=r, num=0)
                _ = self.gen_diff_var(r=r, num=1)
                _ = self.gen_diff_var(r=r, num=2)
                _ = self.gen_diff_var(r=r, num=3)
        self.prob_cnt = self.cnt
        for r in range(end):
            _ = self.gen_prob_var(r=r, num=0)
            _ = self.gen_prob_var(r=r, num=1)
            _ = self.gen_prob_var(r=r, num=2)

        inp0 = self.load_diff_var(r=0, num=0)
        inp1 = self.load_diff_var(r=0, num=1)
        inp2 = self.load_diff_var(r=0, num=2)
        inp3 = self.load_diff_var(r=0, num=3)
        inp = inp0 + inp1 + inp2 + inp3

        if amp == 0:
            add = ''
            for i in range(self.n):
                add += ('%d ' % inp[i])
            add += '0'
            clause += [add]
        for r in range(end):
            add_clause, out = self.lea_diff_one_round(inp=inp, r=r)
            clause += add_clause
            inp = out
        return clause

    def cardinality_constraints_node(self, print_node):
        clause = []
        if print_node.left == None:
            return clause

        Left = print_node.left
        Right = print_node.right
        m1 = Left.nbr
        m2 = Right.nbr

        for i in range(1, m1 + 1):
            for j in range(1, m2 + 1):
                clause += ["-%d -%d %d 0" % (Left.fr + i - 1, Right.fr + j - 1, print_node.fr + i + j - 1)]
                clause += ["%d %d -%d 0" % (Left.fr + i - 1, Right.fr + j - 1, print_node.fr + i + j - 2)]
        for j in range(m2):
            clause += ["-%d %d 0" % (Right.fr + j, print_node.fr + j)]
            clause += ["%d -%d 0" % (Right.fr + j, print_node.fr + m1 + j)]
        for i in range(m1):
            clause += ["-%d %d 0" % (Left.fr + i, print_node.fr + i)]
            clause += ["%d -%d 0" % (Left.fr + i, print_node.fr + m2 + i)]

        clause += self.cardinality_constraints_node(print_node.left)
        clause += self.cardinality_constraints_node(print_node.right)

        return clause

    def cardinality_constraints(self, root, upper):
        clause = []
        clause += self.cardinality_constraints_node(root)
        for i in range(upper, root.nbr):
            clause += ["-%d 0" %(root.fr + i)]
        return clause

    def diff_make_cnf(self, Nr, prob_val, model_name):
        clause = []

        clause += self.lea_diff_clause(end=Nr)

        word_size = self.w

        Npro = (word_size - 1) * 3 * Nr
        proVar = [None] * Npro
        for i in range(Npro):
            proVar[i] = self.cnt + i

        tree = Tree(fromCnt=0, ExCnt=0, proVar=proVar)
        root = tree.create_tree(fr=proVar[Npro - 1] + 1, nbr=Npro)
        clause += self.cardinality_constraints(root=root, upper=prob_val)

        f = open(model_name, 'w')
        for c in clause:
            f.write(c)
            f.write("\n")
        f.close()
        return clause, self.cnt, self.prob_cnt

    def diff_make_cnf_inp_fix(self, Nr, prob_val, model_name, inp_val):
        clause = []

        clause += self.lea_diff_clause(end=Nr, amp=1)

        word_size = self.w

        Npro = (word_size - 1) * 3 * Nr
        proVar = [None] * Npro
        for i in range(Npro):
            proVar[i] = self.cnt + i

        tree = Tree(fromCnt=0, ExCnt=0, proVar=proVar)
        root = tree.create_tree(fr=proVar[Npro - 1] + 1, nbr=Npro)
        clause += self.cardinality_constraints(root=root, upper=prob_val)

        inp0 = self.load_diff_var(r=0, num=0)
        inp1 = self.load_diff_var(r=0, num=1)
        inp2 = self.load_diff_var(r=0, num=2)
        inp3 = self.load_diff_var(r=0, num=3)

        inp_val0 = inp_val[0]
        inp_val1 = inp_val[1]
        inp_val2 = inp_val[2]
        inp_val3 = inp_val[3]

        mask = 1
        for i in range(word_size):
            tmp0 = inp_val0 & mask
            inp_val0 = inp_val0 >> 1

            if tmp0 == 1:
                clause += ["%d 0" % (inp0[i])]
            else:
                clause += ["-%d 0" % (inp0[i])]

        for i in range(word_size):
            tmp1 = inp_val1 & mask
            inp_val1 = inp_val1 >> 1

            if tmp1 == 1:
                clause += ["%d 0" % (inp1[i])]
            else:
                clause += ["-%d 0" % (inp1[i])]

        for i in range(word_size):
            tmp2 = inp_val2 & mask
            inp_val2 = inp_val2 >> 1

            if tmp2 == 1:
                clause += ["%d 0" % (inp2[i])]
            else:
                clause += ["-%d 0" % (inp2[i])]

        for i in range(word_size):
            tmp3 = inp_val3 & mask
            inp_val3 = inp_val3 >> 1

            if tmp3 == 1:
                clause += ["%d 0" % (inp3[i])]
            else:
                clause += ["-%d 0" % (inp3[i])]

        f = open(model_name, 'w')
        for c in clause:
            f.write(c)
            f.write("\n")
        f.close()
        return clause, self.cnt, self.prob_cnt

    def diff_make_cnf_out_fix(self, Nr, prob_val, model_name, out_val):
        clause = []

        clause += self.lea_diff_clause(end=Nr, amp=1)

        word_size = self.w

        Npro = (word_size - 1) * 3 * Nr
        proVar = [None] * Npro
        for i in range(Npro):
            proVar[i] = self.cnt + i

        tree = Tree(fromCnt=0, ExCnt=0, proVar=proVar)
        root = tree.create_tree(fr=proVar[Npro - 1] + 1, nbr=Npro)
        clause += self.cardinality_constraints(root=root, upper=prob_val)

        out0 = self.load_diff_var(r=Nr, num=0)
        out1 = self.load_diff_var(r=Nr, num=1)
        out2 = self.load_diff_var(r=Nr, num=2)
        out3 = self.load_diff_var(r=Nr - 1, num=0)

        out_val0 = out_val[0]
        out_val1 = out_val[1]
        out_val2 = out_val[2]
        out_val3 = out_val[3]

        mask = 1
        for i in range(word_size):
            tmp0 = out_val0 & mask
            out_val0 = out_val0 >> 1

            if tmp0 == 1:
                clause += ["%d 0" % (out0[i])]
            else:
                clause += ["-%d 0" % (out0[i])]

        for i in range(word_size):
            tmp1 = out_val1 & mask
            out_val1 = out_val1 >> 1

            if tmp1 == 1:
                clause += ["%d 0" % (out1[i])]
            else:
                clause += ["-%d 0" % (out1[i])]

        for i in range(word_size):
            tmp2 = out_val2 & mask
            out_val2 = out_val2 >> 1

            if tmp2 == 1:
                clause += ["%d 0" % (out2[i])]
            else:
                clause += ["-%d 0" % (out2[i])]

        for i in range(word_size):
            tmp3 = out_val3 & mask
            out_val3 = out_val3 >> 1

            if tmp3 == 1:
                clause += ["%d 0" % (out3[i])]
            else:
                clause += ["-%d 0" % (out3[i])]

        f = open(model_name, 'w')
        for c in clause:
            f.write(c)
            f.write("\n")
        f.close()
        return clause, self.cnt, self.prob_cnt 
