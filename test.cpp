#include <iostream>
#include <stdio.h>
#include "MATRIX.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif 

using namespace std;
using namespace cl;

int main() {
	std::size_t size;
	char* str;

	cl_int clGetPlatformIDs(
		cl_uint num_entries,
		cl_platform_id * platforms,
		cl_uint * num_platforms);

	cl_uint nP;
	cl_uint status = clGetPlatformIDs(0, NULL, &nP);
	cl_platform_id* pfs = new cl_platform_id[nP];
	status = clGetPlatformIDs(nP, pfs, NULL);
			
	cl_int clGetPlatformInfo(
		cl_platform_id platform,
		cl_platform_info prm_name,	
		std::size_t prm_value_size,
		void* prm_value,
		std::size_t * prm_value_size_ret);

}