#include <chrono>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void display(const char* descriptor, const thrust::host_vector<int>& arr, const int size, const int display_flag)
{
	if (display_flag)
	{
		printf("%5s: [", descriptor);
		for (int i = 0; i < size; ++i)
		{
			printf("%2d", arr[i]);
			if (i < size - 1)
			{
				printf(", ");
			}
		}
		printf("]\n");
	}
}

int main(int argc, char** argv)
{
	// Read command line arguments
	int dim = 3;
	int display_flag = 0;
	if (argc >= 2)
	{
		dim = atoi(argv[1]);
	}
	if (argc >= 3)
	{
		display_flag = atoi(argv[2]);
	}
	else if(dim < 10)
	{
		display_flag = 1;
	}

	const auto start_overall = std::chrono::high_resolution_clock::now();

	// Initialize vectors	
	thrust::host_vector<int> a(dim);
	thrust::host_vector<int> b(dim);
	thrust::host_vector<int> r(dim);
	thrust::generate(a.begin(), a.end(), rand);
	thrust::generate(b.begin(), b.end(), rand);
	display("a", a, dim, display_flag);
	display("b", b, dim, display_flag);

	// Transfer to device
	thrust::device_vector<int> da = a;
	thrust::device_vector<int> db = b;
	thrust::device_vector<int> dr = r;

	// Add
	auto start = std::chrono::high_resolution_clock::now();
	thrust::transform(da.begin(), da.end(), db.begin(), dr.begin(), thrust::plus<int>());
	thrust::copy(dr.begin(), dr.end(), r.begin());
	auto stop = std::chrono::high_resolution_clock::now();
	display("r", r, dim, display_flag);
	std::cout << "  add: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds\n";

	// Subtract
	start = std::chrono::high_resolution_clock::now();
	thrust::transform(da.begin(), da.end(), db.begin(), dr.begin(), thrust::minus<int>());
	thrust::copy(dr.begin(), dr.end(), r.begin());
	stop = std::chrono::high_resolution_clock::now();
	display("r", r, dim, display_flag);
	std::cout << "  sub: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds\n";

	// Multiply
	start = std::chrono::high_resolution_clock::now();
	thrust::transform(da.begin(), da.end(), db.begin(), dr.begin(), thrust::multiplies<int>());
	thrust::copy(dr.begin(), dr.end(), r.begin());
	stop = std::chrono::high_resolution_clock::now();
	display("r", r, dim, display_flag);
	std::cout << " mult: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds\n";

	// Modulo
	start = std::chrono::high_resolution_clock::now();
	thrust::transform(da.begin(), da.end(), db.begin(), dr.begin(), thrust::modulus<int>());
	thrust::copy(dr.begin(), dr.end(), r.begin());
	stop = std::chrono::high_resolution_clock::now();
	display("r", r, dim, display_flag);
	std::cout << "  mod: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds\n";

	const auto stop_overall = std::chrono::high_resolution_clock::now();
	std::cout << "  all: " << std::chrono::duration_cast<std::chrono::microseconds>(stop_overall - start_overall).count() << " microseconds\n";

	return 0;
}
