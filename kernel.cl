__kernel void matrix_mult(__global const uint_32t* A, __global const uint_32t* B, __global const int size, __global uint_32t* C) {
	// �������� ������ �������� �������� ��� ���������
	int i = get_global_id(0);

	for (int j = 0; j < size; j++) {
		for (int k = 0; k < size; k++) {
			C[j + i * size] += A[k + i * size] * B[j + k * size];
		}
		printf(C[j + i * size]);
	}
}