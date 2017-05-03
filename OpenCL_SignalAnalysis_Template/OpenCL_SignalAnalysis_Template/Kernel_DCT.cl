//////////////////////////////////////////////////////////////////////////////////////////////////
#define PI_ 3.141592653589793
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void computeDCT1D(__global float * input_mat,
                           __global float * ret_mat,
                           int     input_dim)
{
    int   i;
    float c;
    float angle;
    
    i = get_global_id(0);
    c = 0.0;
    
    for (int k = 0; k < input_dim; k += 1)
    {
        angle = (PI_ * ((float)(i * (2 * k + 1))/ (float)(2 * input_dim)));
        c    += (cos(angle)) * input_mat[k];
    }
    
    c *= sqrt(2.0/(float)input_dim);
    
    ret_mat[i] = c;
}

__kernel void computeDCT2D(__global float * input_mat,
                           __global float * ret_mat,
                           int       input_mat_dim_x,
                           int       input_mat_dim_y)
{
    float cu;
    float cv;
    float z;
    
    int v = get_global_id(0);
    int u = get_global_id(1);
    
    cv = (v == 0) ? 1/sqrt(2.0) : 1;
    cu = (u == 0) ? 1/sqrt(2.0) : 1;
    z  = 0;
    for(int y = 0; y < input_mat_dim_y; y += 1)
    {
        for(int x = 0; x < input_mat_dim_x; x += 1)
        {
            float angle_u;
            float angle_v;
            
            angle_u = (u*PI_)*((float)(2*y+1)/(float)(2*input_mat_dim_x));
            angle_v = (v*PI_)*((float)(2*x+1)/(float)(2*input_mat_dim_x));
            
            z += input_mat[x + input_mat_dim_y*y] * cos(angle_v) * cos(angle_u);
        }
    }
    ret_mat[u + input_mat_dim_y*v] = 0.25*cu*cv*z;
}

__kernel void computeIDCT1D(__global float * input_mat,
                            __global float * ret_mat,
                            int     input_dim)
{
    int   i;
    float c;
    float angle;
    
    i = get_global_id(0);
    
    c = input_mat[0] / 2.0;
    
    for(int k = 1; k < input_dim; k += 1)
    {
        angle = PI_ * (((float)((2 * i + 1) * k))/((float)(2 * input_dim)));
        c    += (cos(angle)) * input_mat[k];
    }
    
    c *= sqrt(2.0/(float)input_dim);
    
    ret_mat[i] = c;
}

__kernel void computeIDCT2D(__global float * input_mat,
                            __global float * ret_mat,
                            int       input_mat_dim_x,
                            int       input_mat_dim_y)
{
    int y = get_global_id(0);
    int x = get_global_id(1);
    
    float z  = 0;
    
    for(int v = 0; v < input_mat_dim_y; v += 1)
    {
        for(int u = 0; u < input_mat_dim_x; u += 1)
        {
            float angle_u;
            float angle_v;
            
            float cv;
            float cu;
            
            cv = (v == 0) ? 1/sqrt(2.0) : 1;
            cu = (u == 0) ? 1/sqrt(2.0) : 1;
            
            
            angle_u = (u*PI_)*((float)(2*x+1)/(float)(2*input_mat_dim_x));
            angle_v = (v*PI_)*((float)(2*y+1)/(float)(2*input_mat_dim_x));
            
            z += cv * cu * input_mat[u + input_mat_dim_x*v] * cos(angle_v) * cos(angle_u);
        }
    }
    
    z /= 4.0;
    
    if(z > 255.0)  { z = 255.0;       }
    if(z < 0)      { z = 0;           }
    
    ret_mat[y + input_mat_dim_y*x] = z;
}