/* The coder of shape/hampack.py: points of A_L repeated over a block of n weights, cut by a ball.
 *
 * A_L = C + 2Z^L with C the extended Hamming code of length L = 2^m: a point is a vector of whole numbers
 * whose parities, block of L by block of L, form words of the code. Coordinate j of a block has the
 * check vector h[j] = j | L; the XOR of the check vectors of the odd entries read so far is the state.
 *
 * Numbering: the entries are read in order with the symbols 0, 1, -1, 2, -2, ...; the number of a point
 * is the count of the points that come before it. ways[(i * states + s) * (bound + 1) + r] is the count of
 * ways to finish a point from entry i in state s with squared length at most r, every block ending in
 * state 0. At the first entry of a block the rows of states other than 0 are zero. The table is built by
 * hampack.py in whole-number arithmetic and handed over as limbs of 64 bits. Same arithmetic as
 * ballpack.c, with a state.
 *
 * hp_closest is the nearest point of A_L to each block of L, the dynamic program of
 * experiments/hamming-lattices-real-weights-2026-11-24/hamming_dp.c, copied.
 *
 * Build:  cc -O2 -ffp-contract=off -shared -fPIC -o hampack.so shape/hampack.c -lm
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LIMBS 3

typedef struct { uint64_t w[LIMBS]; } num;

static int g_n = 0, g_L = 0, g_states = 0, g_bound = 0, g_bytes = 0;
static int g_h[64];
static num *g_ways = NULL;

static inline int less(const num *a, const num *b) {
    for (int i = LIMBS - 1; i >= 0; --i)
        if (a->w[i] != b->w[i]) return a->w[i] < b->w[i];
    return 0;
}

static inline void add(num *a, const num *b) {
    unsigned __int128 c = 0;
    for (int i = 0; i < LIMBS; ++i) {
        c += (unsigned __int128) a->w[i] + b->w[i];
        a->w[i] = (uint64_t) c;
        c >>= 64;
    }
}

static inline void sub(num *a, const num *b) {      /* a >= b */
    unsigned __int128 borrow = 0;
    for (int i = 0; i < LIMBS; ++i) {
        unsigned __int128 d = (unsigned __int128) a->w[i] - b->w[i] - borrow;
        a->w[i] = (uint64_t) d;
        borrow = (d >> 64) ? 1 : 0;
    }
}

static inline const num *row(int i, int s) {
    return g_ways + ((size_t) i * (size_t) g_states + (size_t) s) * (size_t) (g_bound + 1);
}

/* table: (n + 1) * states * (bound + 1) numbers. Returns 0 if it was taken. */
int hp_init(int n, int L, int bound, int bytes, const uint64_t *table) {
    if (L != 8 && L != 16 && L != 32) return 1;
    if (n < L || n % L || n > 4096 || bound < 0 || bytes < 1 || bytes > 8 * LIMBS) return 1;
    int states = 2 * L;
    size_t count = (size_t) (n + 1) * (size_t) states * (size_t) (bound + 1);
    num *t = (num *) malloc(count * sizeof(num));
    if (!t) return 2;
    memcpy(t, table, count * sizeof(num));
    free(g_ways);
    g_ways = t; g_n = n; g_L = L; g_states = states; g_bound = bound; g_bytes = bytes;
    for (int j = 0; j < L; ++j) g_h[j] = j | L;
    return 0;
}

/* Y: groups * n entries. out: groups * bytes, least byte first. Returns the count of blocks not in the codebook. */
size_t hp_rank(const int8_t *Y, size_t groups, uint8_t *out) {
    size_t bad = 0;
    const int n = g_n, L = g_L;
    for (size_t g = 0; g < groups; ++g) {
        const int8_t *y = Y + g * (size_t) n;
        num total; memset(&total, 0, sizeof total);
        int left = g_bound, ok = 1, s = 0;
        for (int i = 0; i < n && ok; ++i) {
            const int h = g_h[i % L];
            int v = y[i];
            int a = v < 0 ? -v : v;
            int cost = a * a;
            if (cost > left) { ok = 0; break; }
            if (a > 0) {
                add(&total, row(i + 1, s) + left);                         /* the symbol 0 */
                for (int k = 1; k < a; ++k) {
                    const num *r = row(i + 1, (k & 1) ? s ^ h : s) + (left - k * k);
                    add(&total, r);
                    add(&total, r);
                }
                if (v < 0) add(&total, row(i + 1, (a & 1) ? s ^ h : s) + (left - cost));
            }
            if (a & 1) s ^= h;
            left -= cost;
            if ((i + 1) % L == 0) {                                         /* a block of A_L must end in state 0 */
                if (s) { ok = 0; break; }
            }
        }
        uint8_t *o = out + g * (size_t) g_bytes;
        if (!ok) { ++bad; memset(o, 0xff, (size_t) g_bytes); continue; }
        for (int b = 0; b < g_bytes; ++b) o[b] = (uint8_t) (total.w[b / 8] >> (8 * (b % 8)));
    }
    return bad;
}

static inline int unrank_one(const uint8_t *in, int8_t *y) {
    const int n = g_n, L = g_L;
    num k; memset(&k, 0, sizeof k);
    for (int b = 0; b < g_bytes; ++b) k.w[b / 8] |= (uint64_t) in[b] << (8 * (b % 8));
    int left = g_bound, s = 0;
    for (int i = 0; i < n; ++i) {
        const int h = g_h[i % L];
        const num *ways = row(i + 1, s) + left;                            /* the symbol 0 */
        if (less(&k, ways)) { y[i] = 0; continue; }
        sub(&k, ways);
        int found = 0;
        for (int a = 1; a * a <= left; ++a) {
            const int t = (a & 1) ? s ^ h : s;
            ways = row(i + 1, t) + (left - a * a);
            if (less(&k, ways)) { y[i] = (int8_t) a; left -= a * a; s = t; found = 1; break; }
            sub(&k, ways);
            if (less(&k, ways)) { y[i] = (int8_t) -a; left -= a * a; s = t; found = 1; break; }
            sub(&k, ways);
        }
        if (!found) return 1;                                               /* the number is not below the count */
    }
    return 0;
}

/* in: groups * bytes. Y: groups * n entries. Returns the count of numbers that name no point. */
size_t hp_unrank(const uint8_t *in, size_t groups, int8_t *Y) {
    size_t bad = 0;
    for (size_t g = 0; g < groups; ++g) {
        if (unrank_one(in + g * (size_t) g_bytes, Y + g * (size_t) g_n)) {
            ++bad;
            memset(Y + g * (size_t) g_n, 0, (size_t) g_n);
        }
    }
    return bad;
}

/* A whole decode, for timing: records of (1 byte of scale code, bytes of the number) to 32-bit floats. */
size_t hp_decode(const uint8_t *records, size_t groups, double top, int steps, float *out) {
    float scale[256];
    for (int c = 0; c < 256; ++c) scale[c] = (float) (top * pow(2.0, -(double) c / steps));
    size_t bad = 0;
    const int n = g_n;
    int8_t y[4096];
    for (size_t g = 0; g < groups; ++g) {
        const uint8_t *r = records + g * (size_t) (g_bytes + 1);
        if (unrank_one(r + 1, y)) { ++bad; memset(y, 0, (size_t) n); }
        const float sc = scale[r[0]];
        float *o = out + g * (size_t) n;
        for (int i = 0; i < n; ++i) o[i] = sc * (float) y[i];
    }
    return bad;
}

/* Nearest point of A_L (L = 2^m, m = 3, 4, 5) to each of `blocks` blocks of L entries. Copied from
 * hamming_dp.c of the record hamming-lattices-real-weights-2026-11-24, unchanged in what it computes. */
int hp_closest(const double *x, double *out, size_t blocks, int m)
{
    if (m < 3 || m > 5)
        return -1;
    const int n = 1 << m, states = 2 * n;
    double S[64], T[64], e[32], o[32], de[32], dd[32];
    unsigned char take[32][64];
    for (size_t b = 0; b < blocks; b++) {
        const double *xb = x + b * (size_t)n;
        double *yb = out + b * (size_t)n;
        for (int i = 0; i < n; i++) {
            double ev = 2.0 * rint(xb[i] / 2.0);
            double ov = (xb[i] >= ev) ? ev + 1.0 : ev - 1.0;
            double d1 = xb[i] - ev, d2 = xb[i] - ov;
            e[i] = ev; o[i] = ov; de[i] = d1 * d1; dd[i] = d2 * d2;
        }
        for (int s = 0; s < states; s++)
            S[s] = INFINITY;
        S[0] = 0.0;
        for (int i = 0; i < n; i++) {
            const int h = i | n;
            for (int s = 0; s < states; s++) {
                double a = S[s] + de[i];
                double c = S[s ^ h] + dd[i];
                if (c < a) { T[s] = c; take[i][s] = 1; }
                else { T[s] = a; take[i][s] = 0; }
            }
            for (int s = 0; s < states; s++)
                S[s] = T[s];
        }
        int s = 0;
        for (int i = n - 1; i >= 0; i--) {
            if (take[i][s]) { yb[i] = o[i]; s ^= (i | n); }
            else yb[i] = e[i];
        }
    }
    return 0;
}
