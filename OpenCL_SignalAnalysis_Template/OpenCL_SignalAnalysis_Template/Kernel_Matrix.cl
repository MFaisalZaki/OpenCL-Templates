/*
 * Kernel_Matrix will provide all possible mainpulations for a matrix.
 * It is intended to be used as a library for DCT computation.
 */

__kernel void multiplyMatrixAB(__global float * mat_a,
                               __global float * mat_b,
                               __global float * mat_c,
                               int     dima,
                               int     dimb)
{
    int tx = get_global_id(0);
    int ty = get_global_id(1);
    
    float value = 0;
    
    for (int k = 0; k < dima; k += 1)
    {
        value += mat_a[ty * dima + k] * mat_b[k * dimb + tx];
    }
    
    mat_c[ty * dima + tx] = value;
}