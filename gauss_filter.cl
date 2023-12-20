void kernel opencl_processing(global float* gauss, global uchar* input, global uchar* output, int rows, int cols) {
    int i = get_global_id(0);;
    int j = get_global_id(1);;
    int g_size = 3;

    if (i < rows && j < cols) {
        float blur_pixel = 0.0;
        for (int x = -g_size / 2; x <= g_size / 2; x++) {
            for (int y = -g_size / 2; y <= g_size / 2; y++) {
                int current_row = i + x;
                int current_col = j + y;

                if (current_row >= 0 && current_row < rows && current_col >= 0 && current_col < cols) {
                    float filter_value = gauss[(y + g_size / 2) * g_size + (x + g_size / 2)];
                    blur_pixel += input[current_row * cols + current_col] * filter_value;
                }
            }
        }
        output[i * cols + j] = blur_pixel;
    }
};
