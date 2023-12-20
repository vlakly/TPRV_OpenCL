void kernel matrix_mult(global const uint* A, global const uint* B, global uint* C, int S) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    uint buf = 0;
    for (int i = 0; i < S; i++) {
        buf += A[y * S + i] * B[x + S * i];
    }
    C[x + y * S] = buf;
};