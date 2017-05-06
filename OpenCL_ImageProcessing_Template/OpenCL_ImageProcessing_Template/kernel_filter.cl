__kernel void Filter(__global    float4  *input,
                     __global    float4  *output,
                     __constant  float   *filter_ws,
                                 float   threshold,
                                 int     filter_size)
{
    
    int2 pos = {get_global_id(0), get_global_id(1)};
    int2 g_size = {get_global_size(0), get_global_size(1)};
    int index = pos.y * g_size.x + pos.x;
    
    int half_filter_size = filter_size/2;
    int filter_i = 0;
    float4 response = (float4)0.0f;
    
    /* Filter can overlap with image boundary when index is between the following ranges:
     *  a. index % get_global_size(0) == 0.
     *  b. index ??
     */
    if(   (0 != (index % g_size.x))                 /* First vertical line.    */
       && (index >= g_size.x)                       /* First horizontal line.  */
       && (index <= (g_size.x*g_size.y - g_size.x)) /* Second horizontal line. */
       && (index < (pos.y*g_size.x + g_size.x - 1)) /* Second horizontal line. */
       )
    {
        /* Apply filter weights if the filter is not boundary of the image. */
        for(int r = -half_filter_size; r <= half_filter_size; r += 1)
        {
            int current_row = index + r * g_size.x;
            for(int c = -half_filter_size; c <= half_filter_size; c += 1)
            {
                response += input[current_row + c] * (float4)filter_ws[filter_i];
                filter_i += 1;
            }
        }
        
    }
    else
    {
        /* Start padding the filter but we need to know which corner are we in ? */
        output[index] = 0;
    }
    
    /* Check compare threshold. */
    output[index] = (response > (float4)threshold) ? (float4)255 : (float4)0;
}
