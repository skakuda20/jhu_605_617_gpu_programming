__kernel void convolution(__global const uchar4* input,
                          __global uchar4* output,
                          __global const float* mask,
                          const int maskWidth,
                          const int width,
                          const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int halfWidth = maskWidth / 2;

    float4 sum = (float4)(0.0f);

    for (int j = -halfWidth; j <= halfWidth; j++) {
        for (int i = -halfWidth; i <= halfWidth; i++) {
            int px = clamp(x + i, 0, width - 1);
            int py = clamp(y + j, 0, height - 1);
            uchar4 pixel = input[py * width + px];
            float w = mask[(j + halfWidth) * maskWidth + (i + halfWidth)];
            sum += convert_float4(pixel) * w;
        }
    }

    sum = clamp(sum, 0.0f, 255.0f);
    output[y * width + x] = convert_uchar4(sum);
}
