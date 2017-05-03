__kernel void Filter(__global    float4  *input,
                     __global    float4  *output,
                     __constant  float  *filter_ws,
                     int filter_size)
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
        output[index] = 255;
        
    }
    output[index] = response;
}

/*
 
 int fIndex = 0;
 float4 sum = (float4) 0.0;
 
 for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
 {
	int curRow = my + r * IMAGE_W;
	for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++)
	{
 sum += input[ curRow + c ] * filter[ fIndex ];
 fIndex++;
	
	}
 }
 output[my] = sum;
 
 
 
 __kernel void convolute(
	const __global float * input,
	__global float * output,
	__global float * filter
 )
 {
 
	int rowOffset = get_global_id(1) * IMAGE_W * 4;
	int my = 4 * get_global_id(0) + rowOffset;
	
	int fIndex = 0;
	float sumR = 0.0;
	float sumG = 0.0;
	float sumB = 0.0;
	float sumA = 0.0;
	
 
	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
	{
 int curRow = my + r * (IMAGE_W * 4);
 for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex += 4)
 {
 int offset = c * 4;
 
 sumR += input[ curRow + offset   ] * filter[ fIndex   ];
 sumG += input[ curRow + offset+1 ] * filter[ fIndex+1 ];
 sumB += input[ curRow + offset+2 ] * filter[ fIndex+2 ];
 sumA += input[ curRow + offset+3 ] * filter[ fIndex+3 ];
 }
	}
	
	output[ my     ] = sumR;
	output[ my + 1 ] = sumG;
	output[ my + 2 ] = sumB;
	output[ my + 3 ] = sumA;
	
 }
 
 
 
 */
