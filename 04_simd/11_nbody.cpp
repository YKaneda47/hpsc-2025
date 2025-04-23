#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);

    __m512 xj = _mm512_load_ps(x);
    __m512 yj = _mm512_load_ps(y);
    __m512 mj = _mm512_load_ps(m);

    __m512 rx = _mm512_sub_ps(xi, xj);
    __m512 ry = _mm512_sub_ps(yi, yj); 
    __m512 r = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));
    __m512 rinv = _mm512_rsqrt14_ps(r);
    __m512 rinv_cubed = _mm512_mul_ps(_mm512_mul_ps(rinv, rinv), rinv);
    __m512 coeff = _mm512_mul_ps(mj, rinv_cubed);

    __m512i idx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    __mmask16 mask = _mm512_cmp_epi32_mask(idx, _mm512_set1_epi32(i), _MM_CMPINT_NE);

    __m512 fx_vec = _mm512_mask_mul_ps(_mm512_setzero_ps(), mask, rx, coeff);
    __m512 fy_vec = _mm512_mask_mul_ps(_mm512_setzero_ps(), mask, ry, coeff);
    
    float fx_sum = 0.0;
    float fy_sum = 0.0;
    fx_sum = _mm512_reduce_add_ps(fx_vec);
    fy_sum = _mm512_reduce_add_ps(fy_vec);

    fx[i] = -fx_sum;
    fy[i] = -fy_sum;
    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
