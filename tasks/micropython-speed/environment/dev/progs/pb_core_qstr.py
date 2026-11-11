# This tests qstr_find_strn() speed when the string being searched for is not found.


def test(r):
    for _ in r:
        str("a string that shouldn't be interned")


###########################################################################
# Benchmark interface

bm_params = {
    (32, 10): (400,),
    (1000, 10): (4000,),
    (5000, 10): (40000,),
}


def bm_setup(params):
    (nloop,) = params
    return lambda: test(range(nloop)), lambda: (nloop // 100, None)


# Scaled for this task: run() repeats REPEAT times (upstream runs it once).
REPEAT = 4


def _bench_main(N=1000, M=100):
    cur_nm = (0, 0)
    param = None
    for nm, p in bm_params.items():
        if 10 * nm[0] <= 12 * N and nm[1] <= M and nm > cur_nm:
            cur_nm = nm
            param = p
    run, result = bm_setup(param)
    for _ in range(REPEAT):
        run()
    norm, out = result()
    print(cur_nm, norm, out)


_bench_main()
