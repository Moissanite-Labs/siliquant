/* Nearest point of A_n = C + 2Z^n, C the extended Hamming code of length n = 2^m (m = 3, 4, 5).
 *
 * The same dynamic program as hamming.closest in NumPy, step for step, so that the two agree bit for bit:
 * every coordinate takes its nearest even or nearest odd whole number; the parity bits must form a word of
 * the code; the cheapest choice is found over the 2^(m+1) partial syndromes. Coordinate i has the check
 * vector i | n. Ties: a move is taken only when strictly cheaper, as in the NumPy code.
 *
 * Built with -ffp-contract=off so that no multiply and add are fused, which would change the last bit.
 */
#include <math.h>
#include <stddef.h>

int hamming_closest(const double *x, double *out, size_t blocks, int m)
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
