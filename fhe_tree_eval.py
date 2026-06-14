"""
Encrypted decision-tree evaluation primitives.

Ported from fhe-decision-tree/CKKS-tree-eval.py (ApproxComp + EncTreeEvaluator).
"""


def approx_comp(c, cc):
    c2 = cc.EvalMult(c, c)
    cf = [
        7.58234312199665e-06,
        -0.000103379119789856,
        0.0011064732653286301,
        -0.00815025951047975,
        0.0422464691571231,
        -0.157615263376308,
        0.429363446968902,
        -0.859081774093833,
        1.25888200889262,
        -1.33313518631642,
        0.991778115478544,
        -0.491174546890581,
        0.145282744849388,
        -0.0194054317213377,
    ]
    compcf = [value / 2 for value in cf]
    r1 = cc.EvalPolyPS(c2, compcf)
    r2 = cc.EvalMult(c, r1)
    scaledresult = cc.EvalMult(10e5, r2)
    return cc.EvalAdd(scaledresult, 0.5)


def enc_tree_evaluator(c, cc, num_trees, num_slots, tree_depth, public_key):
    num_nodes = 2**tree_depth - 1

    onebarciphers = []
    cvals = []

    for i in range(num_nodes):
        total_one_bar = []
        total_one_vec = []
        bl = (i + 1).bit_length() - 1
        num_ones = int(2 ** (tree_depth - bl - 1))
        treeposition = int((i + 1) % (2**bl) * 2 ** (tree_depth - bl))
        one_vec_i = [0 for _ in range(num_nodes + 1)]
        one_bar_i = [0 for _ in range(num_nodes + 1)]
        one_vec_i[i] = 1

        offset = int(2 ** (tree_depth - bl - 1))
        for j in range(treeposition + offset, treeposition + offset + num_ones):
            one_bar_i[j] = 1

        for _ in range(num_trees):
            total_one_vec += one_vec_i
            total_one_bar += one_bar_i

        encr_one_vec = cc.Encrypt(public_key, cc.MakeCKKSPackedPlaintext(total_one_vec))
        encr_one_bar = cc.Encrypt(public_key, cc.MakeCKKSPackedPlaintext(total_one_bar))

        cvals.append(cc.EvalMult(encr_one_vec, c))
        onebarciphers.append(encr_one_bar)

    onehotvals = []
    for i in range(num_nodes):
        c_next = cvals[i]
        bl = (i + 1).bit_length() - 1
        firstrotval = -int((i + 1) % (2**bl) * 2 ** (tree_depth - bl) - i)

        if firstrotval != 0:
            c_next = cc.EvalRotate(c_next, firstrotval)

        for j in range(tree_depth - 1 - bl):
            c_rot = cc.EvalRotate(c_next, int(-(2**j)))
            c_next = cc.EvalAdd(c_rot, c_next)

        comprotval = int(-(2 ** (tree_depth - bl - 1)))
        c_subtract = cc.EvalRotate(c_next, comprotval)
        c_complement = cc.EvalSub(onebarciphers[i], c_subtract)
        c_semifinal = cc.EvalAdd(c_next, c_complement)
        onehotvals.append(c_semifinal)

    return cc.EvalAddMany(onehotvals)
