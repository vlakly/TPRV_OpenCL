__kernel void vector_add(__global uint_32t* A, __global uint_32t* B, __global uint_32t* C) {
	// Получить индекс текущего элемента для обработки
	int i = get_global_id(0);

	C[i] = A[i] + B[i];
}