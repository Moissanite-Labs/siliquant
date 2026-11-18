/* Numbers the points of Z^n inside a ball, for whole matrices at a time: the coder of shape/ballpack.py.
 *
 * The same numbering as shape/ballindex.py for the lattice "Z8": read the n entries in order, with the
 * symbols of an entry in the order 0, 1, -1, 2, -2, ...; the number of a point is the count of the
 * points that come before it. The table of counts is built by ballindex.py in whole-number arithmetic
 * and handed to zb_init as limbs of 64 bits, so this file holds no arithmetic that could build it wrongly.
 *
 * A number has at most 64 * LIMBS bits. With LIMBS = 3 that is 192, enough for the 136 bits of a block
 * of 32 weights; ballpack.py checks the size of the codebook before it calls.
 *
 * Build:  cc -O2 -shared -fPIC -o ballpack.so shape/ballpack.c
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LIMBS 3

typedef struct { uint64_t w[LIMBS]; } num;

static int g_n = 0;            /* weights in a block */
static int g_bound = 0;        /* largest squared length */
static int g_bytes = 0;        /* bytes of a stored number */
static num *g_ways = NULL;     /* ways[m * (bound + 1) + r]: points of Z^m of squared length at most r */

static inline int less(const num *a, const num *b) {
    for (int i = LIMBS - 1; i >= 0; --i) {
        if (a->w[i] != b->w[i]) return a->w[i] < b->w[i];
    }
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

/* table: (n + 1) * (bound + 1) numbers of LIMBS limbs each, least limb first. Returns 0 if it was taken. */
int zb_init(int n, int bound, int bytes, const uint64_t *table) {
    if (n < 1 || n > 4096 || bound < 0 || bytes < 1 || bytes > 8 * LIMBS) return 1;
    size_t count = (size_t) (n + 1) * (size_t) (bound + 1);
    num *t = (num *) malloc(count * sizeof(num));
    if (!t) return 2;
    memcpy(t, table, count * sizeof(num));
    free(g_ways);
    g_ways = t; g_n = n; g_bound = bound; g_bytes = bytes;
    return 0;
}

/* Y: groups * n entries. out: groups * bytes, each number least byte first. Returns the count of blocks not in the codebook. */
size_t zb_rank(const int8_t *Y, size_t groups, uint8_t *out) {
    size_t bad = 0;
    const int n = g_n, B = g_bound;
    for (size_t g = 0; g < groups; ++g) {
        const int8_t *y = Y + g * (size_t) n;
        num total; memset(&total, 0, sizeof total);
        int left = B, ok = 1;
        for (int i = 0; i < n && ok; ++i) {
            const num *row = g_ways + (size_t) (n - i - 1) * (size_t) (B + 1);
            int v = y[i];
            int cost = v * v;
            if (cost > left) { ok = 0; break; }
            /* the symbols before v: 0, then 1, -1, ..., then for a negative v also +|v| */
            int a = v < 0 ? -v : v;
            if (a > 0) {
                add(&total, row + left);                       /* the symbol 0 */
                for (int k = 1; k < a; ++k) {
                    int c = k * k;                              /* c < cost <= left */
                    add(&total, row + (left - c));
                    add(&total, row + (left - c));
                }
                if (v < 0) add(&total, row + (left - cost));    /* +|v| comes before -|v| */
            }
            left -= cost;
        }
        uint8_t *o = out + g * (size_t) g_bytes;
        if (!ok) { ++bad; memset(o, 0xff, (size_t) g_bytes); continue; }
        for (int b = 0; b < g_bytes; ++b) o[b] = (uint8_t) (total.w[b / 8] >> (8 * (b % 8)));
    }
    return bad;
}

static inline int unrank_one(const uint8_t *in, int8_t *y) {
    const int n = g_n, B = g_bound;
    num k; memset(&k, 0, sizeof k);
    for (int b = 0; b < g_bytes; ++b) k.w[b / 8] |= (uint64_t) in[b] << (8 * (b % 8));
    int left = B;
    for (int i = 0; i < n; ++i) {
        const num *row = g_ways + (size_t) (n - i - 1) * (size_t) (B + 1);
        const num *ways = row + left;                           /* the symbol 0 */
        if (less(&k, ways)) { y[i] = 0; continue; }
        sub(&k, ways);
        int found = 0;
        for (int a = 1; a * a <= left; ++a) {
            ways = row + (left - a * a);
            if (less(&k, ways)) { y[i] = (int8_t) a; left -= a * a; found = 1; break; }
            sub(&k, ways);
            if (less(&k, ways)) { y[i] = (int8_t) -a; left -= a * a; found = 1; break; }
            sub(&k, ways);
        }
        if (!found) return 1;                                   /* the number is not below the size of the codebook */
    }
    return 0;
}

/* in: groups * bytes. Y: groups * n entries. Returns the count of numbers that name no point. */
size_t zb_unrank(const uint8_t *in, size_t groups, int8_t *Y) {
    size_t bad = 0;
    for (size_t g = 0; g < groups; ++g) {
        if (unrank_one(in + g * (size_t) g_bytes, Y + g * (size_t) g_n)) {
            ++bad;
            memset(Y + g * (size_t) g_n, 0, (size_t) g_n);
        }
    }
    return bad;
}

/* A whole decode, for timing: records of (1 byte of scale code, bytes of the number) to 32-bit floats.
 * The scale of a block is top * 2^(-code / steps). Returns the count of numbers that name no point. */
size_t zb_decode(const uint8_t *records, size_t groups, double top, int steps, float *out) {
    float scale[256];
    for (int c = 0; c < 256; ++c) scale[c] = (float) (top * pow(2.0, -(double) c / steps));
    size_t bad = 0;
    const int n = g_n;
    int8_t y[4096];
    for (size_t g = 0; g < groups; ++g) {
        const uint8_t *r = records + g * (size_t) (g_bytes + 1);
        if (unrank_one(r + 1, y)) { ++bad; memset(y, 0, (size_t) n); }
        const float s = scale[r[0]];
        float *o = out + g * (size_t) n;
        for (int i = 0; i < n; ++i) o[i] = s * (float) y[i];
    }
    return bad;
}
