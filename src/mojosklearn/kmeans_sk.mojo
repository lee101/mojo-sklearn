"""K-means that reproduces scikit-learn (1.9) exactly: MT19937 RandomState, greedy k-means++ (2 + ln k trials), centered Lloyd with
relative tol, empty-cluster relocation, strict-convergence exit, final E-step, best-of-n_init with the same-clustering rule."""

from std.math import sqrt, log

from mojosklearn.linalg import Ptr

comptime MT_N = 624
comptime MT_M = 397
comptime W = 4


def pptr(l: List[Float64]) -> Ptr:
    return Ptr(unsafe_from_address=Int(l.unsafe_ptr()))


struct MT19937(Movable):
    var mt: List[UInt32]
    var idx: Int

    def __init__(out self, seed: Int):
        self.mt = List[UInt32](length=MT_N, fill=0)
        var s = UInt32(seed & 0xFFFFFFFF)
        for pos in range(MT_N):
            self.mt[pos] = s
            s = UInt32(1812433253) * (s ^ (s >> 30)) + UInt32(pos + 1)
        self.idx = MT_N

    def _refill(mut self):
        comptime UP = UInt32(0x80000000)
        comptime LO = UInt32(0x7FFFFFFF)
        comptime MAG = UInt32(0x9908B0DF)
        for kk in range(MT_N - MT_M):
            var y = (self.mt[kk] & UP) | (self.mt[kk + 1] & LO)
            self.mt[kk] = self.mt[kk + MT_M] ^ (y >> 1) ^ (MAG if (y & 1) != 0 else UInt32(0))
        for kk in range(MT_N - MT_M, MT_N - 1):
            var y = (self.mt[kk] & UP) | (self.mt[kk + 1] & LO)
            self.mt[kk] = self.mt[kk + MT_M - MT_N] ^ (y >> 1) ^ (MAG if (y & 1) != 0 else UInt32(0))
        var y = (self.mt[MT_N - 1] & UP) | (self.mt[0] & LO)
        self.mt[MT_N - 1] = self.mt[MT_M - 1] ^ (y >> 1) ^ (MAG if (y & 1) != 0 else UInt32(0))
        self.idx = 0

    def next_u32(mut self) -> UInt32:
        if self.idx >= MT_N:
            self._refill()
        var y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & UInt32(0x9D2C5680)
        y ^= (y << 15) & UInt32(0xEFC60000)
        y ^= y >> 18
        return y

    def random_sample(mut self) -> Float64:
        var a = Float64(self.next_u32() >> 5)
        var b = Float64(self.next_u32() >> 6)
        return (a * 67108864.0 + b) / 9007199254740992.0


@always_inline
def sqdist(a: Ptr, b: Ptr, d: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= d:
        var t = a.load[width=W](i) - b.load[width=W](i)
        acc += t * t
        i += W
    var s = acc.reduce_add()
    while i < d:
        var t = a[i] - b[i]
        s += t * t
        i += 1
    return s


@always_inline
def dot2(a: Ptr, b: Ptr, d: Int) -> Float64:
    var a0 = SIMD[DType.float64, W](0.0)
    var a1 = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + 2 * W <= d:
        a0 += a.load[width=W](i) * b.load[width=W](i)
        a1 += a.load[width=W](i + W) * b.load[width=W](i + W)
        i += 2 * W
    var s = (a0 + a1).reduce_add()
    while i < d:
        s += a[i] * b[i]
        i += 1
    return s


def _row_dist(x: Ptr, xn: List[Float64], n: Int, d: Int, cand: Int, dst: List[Float64]):
    var c = x + cand * d
    var xx = dot2(c, c, d)
    var o = Ptr(unsafe_from_address=Int(dst.unsafe_ptr()))
    for i in range(n):
        var t = -2.0 * dot2(x + i * d, c, d)
        t += xx
        t += xn[i]
        o[i] = t if t > 0.0 else 0.0


def kmeans_pp_sk(x: Ptr, xn: List[Float64], centers: Ptr, n: Int, d: Int, k: Int, mut rs: MT19937):
    var trials = 2 + Int(log(Float64(k)))
    var p = 1.0 / Float64(n)
    var cdf = List[Float64](length=n, fill=0.0)
    var run = 0.0
    for i in range(n):
        run += p
        cdf[i] = run
    var last = cdf[n - 1]
    var u0 = rs.random_sample()
    var first = n
    for i in range(n):
        if cdf[i] / last > u0:
            first = i
            break
    if first >= n:
        first = n - 1
    for j in range(d):
        centers[j] = x[first * d + j]
    var closest = List[Float64](length=n, fill=0.0)
    _row_dist(x, xn, n, d, first, closest)
    var pot = 0.0
    for i in range(n):
        pot += closest[i]
    var cum = List[Float64](length=n, fill=0.0)
    var cand = List[Int](length=trials, fill=0)
    var dc = List[Float64](length=n, fill=0.0)
    var best_d = List[Float64](length=n, fill=0.0)
    for c in range(1, k):
        var s = 0.0
        for i in range(n):
            s += closest[i]
            cum[i] = s
        for t in range(trials):
            var v = rs.random_sample() * pot
            var lo = 0
            var hi = n
            while lo < hi:
                var mid = (lo + hi) // 2
                if cum[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            cand[t] = min(lo, n - 1)
        var best_pot = 0.0
        var best_t = -1
        for t in range(trials):
            _row_dist(x, xn, n, d, cand[t], dc)
            var tot = 0.0
            for i in range(n):
                var m = closest[i] if closest[i] < dc[i] else dc[i]
                dc[i] = m
                tot += m
            if best_t < 0 or tot < best_pot:
                best_pot = tot
                best_t = t
                for i in range(n):
                    best_d[i] = dc[i]
        pot = best_pot
        for i in range(n):
            closest[i] = best_d[i]
        for j in range(d):
            centers[c * d + j] = x[cand[best_t] * d + j]


def _assign_block[KW: Int](x: Ptr, ct: Ptr, hp: Ptr, kp: Int, best: Ptr, mut bidx: List[Int32], n: Int, d: Int, c0: Int):
    for i in range(n):
        var xi = x + i * d
        var acc = SIMD[DType.float64, KW](0.0)
        for j in range(d):
            acc += SIMD[DType.float64, KW](xi[j]) * (ct + j * kp + c0).load[width=KW](0)
        var v = hp.load[width=KW](c0) - acc
        var bd = best[i]
        var bj = bidx[i]
        comptime for l in range(KW):
            if v[l] < bd:
                bd = v[l]
                bj = Int32(c0 + l)
        best[i] = bd
        bidx[i] = bj


def assign_rows(x: Ptr, centers: Ptr, half: List[Float64], mut labels: List[Int32], n: Int, d: Int, k: Int):
    """Nearest center by argmin(0.5|c|^2 - x.c). Centers are transposed into [d][kp] so one broadcast x[i, j] feeds a whole SIMD row
    of centers (KW = 4 / 8 / 16 / 32 lanes by k); ties resolve to the lowest index."""
    var kw = 4 if k <= 4 else (8 if k <= 8 else (16 if k <= 16 else 32))
    var kp = ((k + kw - 1) // kw) * kw
    var ct = List[Float64](length=d * kp, fill=0.0)
    for c in range(k):
        for j in range(d):
            ct[j * kp + c] = centers[c * d + j]
    var hp = List[Float64](length=kp, fill=1e300)
    for c in range(k):
        hp[c] = half[c]
    var best = List[Float64](length=n, fill=1e300)
    var bidx = List[Int32](length=n, fill=0)
    var c0 = 0
    while c0 < kp:
        if kw == 4:
            _assign_block[4](x, pptr(ct), pptr(hp), kp, pptr(best), bidx, n, d, c0)
        elif kw == 8:
            _assign_block[8](x, pptr(ct), pptr(hp), kp, pptr(best), bidx, n, d, c0)
        elif kw == 16:
            _assign_block[16](x, pptr(ct), pptr(hp), kp, pptr(best), bidx, n, d, c0)
        else:
            _assign_block[32](x, pptr(ct), pptr(hp), kp, pptr(best), bidx, n, d, c0)
        c0 += kw
    for i in range(n):
        labels[i] = bidx[i]


def lloyd_sk(
    x: Ptr, n: Int, d: Int, k: Int, centers: Ptr, mut labels: List[Int32], max_iter: Int, tol: Float64
) -> Tuple[Float64, Int]:
    var cur = List[Float64](length=k * d, fill=0.0)
    var nxt = List[Float64](length=k * d, fill=0.0)
    for i in range(k * d):
        cur[i] = centers[i]
    var half = List[Float64](length=k, fill=0.0)
    var wt = List[Float64](length=k, fill=0.0)
    var shift = List[Float64](length=k, fill=0.0)
    var old = List[Int32](length=n, fill=-1)
    for i in range(n):
        labels[i] = -1
    var strict = False
    var it = 0
    var dist = List[Float64](length=n, fill=0.0)
    for i in range(max_iter):
        it = i + 1
        var cp = pptr(cur)
        var np_ = pptr(nxt)
        for j in range(k):
            half[j] = 0.5 * dot2(cp + j * d, cp + j * d, d)
        assign_rows(x, cp, half, labels, n, d, k)
        for j in range(k * d):
            nxt[j] = 0.0
        for j in range(k):
            wt[j] = 0.0
        for r in range(n):
            var j = Int(labels[r])
            wt[j] += 1.0
            var xr = x + r * d
            var s = np_ + j * d
            var q = 0
            while q + W <= d:
                s.store(q, s.load[width=W](q) + xr.load[width=W](q))
                q += W
            while q < d:
                s[q] += xr[q]
                q += 1
        var nempty = 0
        for j in range(k):
            if wt[j] == 0.0:
                nempty += 1
        if nempty > 0:
            for r in range(n):
                dist[r] = sqdist(x + r * d, cp + Int(labels[r]) * d, d)
            var taken = List[Bool](length=n, fill=False)
            var ei = 0
            for j in range(k):
                if wt[j] != 0.0:
                    continue
                var far = -1
                var fd = -1.0
                for r in range(n):
                    if not taken[r] and dist[r] > fd:
                        fd = dist[r]
                        far = r
                taken[far] = True
                var old_c = Int(labels[far])
                for q in range(d):
                    np_[old_c * d + q] -= x[far * d + q]
                    np_[j * d + q] = x[far * d + q]
                wt[j] = 1.0
                wt[old_c] -= 1.0
                ei += 1
        for j in range(k):
            if wt[j] > 0.0:
                for q in range(d):
                    np_[j * d + q] /= wt[j]
        for j in range(k):
            shift[j] = sqrt(sqdist(np_ + j * d, cp + j * d, d))
        var tmp = cur^
        cur = nxt^
        nxt = tmp^
        var same = True
        for r in range(n):
            if labels[r] != old[r]:
                same = False
                break
        if same:
            strict = True
            break
        var tot = 0.0
        for j in range(k):
            tot += shift[j] * shift[j]
        if tot <= tol:
            break
        for r in range(n):
            old[r] = labels[r]
    var cp = pptr(cur)
    if not strict:
        for j in range(k):
            half[j] = 0.5 * dot2(cp + j * d, cp + j * d, d)
        assign_rows(x, cp, half, labels, n, d, k)
    var inertia = 0.0
    for r in range(n):
        inertia += sqdist(x + r * d, cp + Int(labels[r]) * d, d)
    for i in range(k * d):
        centers[i] = cur[i]
    return (inertia, it)


def _same_clustering(a: List[Int32], b: List[Int32], n: Int, k: Int) -> Bool:
    var m = List[Int](length=k, fill=-1)
    for i in range(n):
        var ai = Int(a[i])
        if m[ai] == -1:
            m[ai] = Int(b[i])
        elif m[ai] != Int(b[i]):
            return False
    return True


def kmeans_fit_sk(
    x_in: Ptr, n: Int, d: Int, k: Int, n_init: Int, max_iter: Int, tol_rel: Float64, seed: Int, centers_out: Ptr, labels_out: Ptr, stats: Ptr
):
    """Writes centers_out[k, d], labels_out[n] (as float64), stats = [inertia, n_iter]."""
    var mean = List[Float64](length=d, fill=0.0)
    for i in range(n):
        for j in range(d):
            mean[j] += x_in[i * d + j]
    for j in range(d):
        mean[j] /= Float64(n)
    var xc = List[Float64](length=n * d, fill=0.0)
    for i in range(n):
        for j in range(d):
            xc[i * d + j] = x_in[i * d + j] - mean[j]
    var x = pptr(xc)
    var xn = List[Float64](length=n, fill=0.0)
    for i in range(n):
        xn[i] = dot2(x + i * d, x + i * d, d)
    var varsum = 0.0
    for j in range(d):
        var s = 0.0
        for i in range(n):
            s += xc[i * d + j] * xc[i * d + j]
        varsum += s / Float64(n)
    var tol = varsum / Float64(d) * tol_rel
    var rs = MT19937(seed)
    var centers = List[Float64](length=k * d, fill=0.0)
    var labels = List[Int32](length=n, fill=0)
    var best_labels = List[Int32](length=n, fill=0)
    var best_centers = List[Float64](length=k * d, fill=0.0)
    var best_inertia = -1.0
    var best_iter = 0
    for _ in range(n_init):
        kmeans_pp_sk(x, xn, pptr(centers), n, d, k, rs)
        var r = lloyd_sk(x, n, d, k, pptr(centers), labels, max_iter, tol)
        if best_inertia < 0.0 or (r[0] < best_inertia and not _same_clustering(labels, best_labels, n, k)):
            best_inertia = r[0]
            best_iter = r[1]
            for i in range(n):
                best_labels[i] = labels[i]
            for i in range(k * d):
                best_centers[i] = centers[i]
    for i in range(k):
        for j in range(d):
            centers_out[i * d + j] = best_centers[i * d + j] + mean[j]
    for i in range(n):
        labels_out[i] = Float64(best_labels[i])
    stats[0] = best_inertia
    stats[1] = Float64(best_iter)
