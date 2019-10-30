// Copyright (c) 2009-2013 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly

// The common code is from:
// https://software.intel.com/en-us/articles/sgemm-for-intel-processor-graphics.

#include <iostream>
#include <ctime>
#include <limits>
#include <cmath>

#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include "basic.hpp"
#include "cmdoptions.hpp"
#include "oclobject.hpp"

using namespace std;

struct OpenGEMMProgram : public OpenCLProgram
{
	cl_kernel kernel;

	// Create and build program and extract kernel.
	// Only one of program_file_name or program_text should be non-empty.
	OpenGEMMProgram(
		OpenCLBasic& oclobjects,
		const std::wstring& program_file_name,
		const string& program_text,
		const string& kernel_name,
		const string& build_options = ""
		);

	~OpenGEMMProgram();

	cl_kernel CreateKernel(const string& kernel_name);

private:

	// Disable copying and assignment to avoid incorrect resource deallocation.
	OpenGEMMProgram(const OpenGEMMProgram&);
	OpenGEMMProgram& operator= (const OpenGEMMProgram&);
};

OpenGEMMProgram::OpenGEMMProgram(
	OpenCLBasic& oclobjects,
	const std::wstring& program_file_name,
	const string& program_text,
	const string& kernel_name,
	const string& build_options
	) :
	OpenCLProgram(oclobjects, program_file_name, program_text, build_options),
	kernel(0)
{
}

OpenGEMMProgram::~OpenGEMMProgram()
{
	try
	{
		if (kernel)
		{
			clReleaseKernel(kernel);
		}
	}
	catch (...)
	{
		destructorException();
	}
}

cl_kernel OpenGEMMProgram::CreateKernel(const string& kernel_name)
{
	using namespace std;

	cl_int err = 0;
	kernel = clCreateKernel(program, kernel_name.c_str(), &err);
	SAMPLE_CHECK_ERRORS(err);
}

int check(int size1, int size3, float* dst, float* cpu_dst)
{
	using namespace std;
	const int n = size1 * size3;
	float err = 0.f;
	for (int i = 0; i < n; ++i)
	{
		//cout << i;
		float localErr = fabs(dst[i] - cpu_dst[i]) / max(fabs(dst[i]), fabs(cpu_dst[i]));
		if (localErr >= 0.001f)
		{
			printf("Error, index %d: Wanted %f, got %f\n", i, cpu_dst[i], dst[i]);
		}
		err = max(fabs(dst[i] - cpu_dst[i]) / max(fabs(dst[i]), fabs(cpu_dst[i])), err);

	}
	return err < 0.001f;
}

void cpuTranspose(
	float * cpu_dst,
	const float * input,
	const int width,
	const int height)
{
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			cpu_dst[i*height + j] = input[j*width + i];
		}
	}
#if 0
	for (int j = 0; j < 20; j++)
		cout << input[j] << ",";
	cout << "\n" << endl;
	for (int j = 0; j < 20; j++)
		cout << cpu_dst[j] << ",";
	cout << "\n" << endl;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cout << input[i*width+j] << ",";
		}
	}
	cout << "\n" << endl;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			cout << cpu_dst[i*height + j] << ",";
		}
	}

#endif
}


void gemm_full(CmdParserGEMM& cmdparser, OpenCLBasic& oclobjects, OpenGEMMProgram& executable, const string& kernel_name, int size1, int size3, size_t group_y, size_t group_x, size_t block_y, size_t block_x)
{
	// Build kernel
	executable.CreateKernel(kernel_name);

	size_t matrix1_memory_size = size1*size3*sizeof(float);
	size_t matrix3_memory_size = size1*size3*sizeof(float);
	size_t alignmentForPtr = zeroCopyPtrAlignment(oclobjects.device);
	size_t alignedSize1 = zeroCopySizeAlignment(matrix1_memory_size, oclobjects.device);
	size_t alignedSize3 = zeroCopySizeAlignment(matrix3_memory_size, oclobjects.device);

	OpenCLDeviceAndHostMemory<float> matrix_A;
	matrix_A.host = (float*)aligned_malloc(alignedSize1, alignmentForPtr);
	if (!matrix_A.host)
	{
		cout << "out of memory\n";
		exit(0);
	}

	OpenCLDeviceAndHostMemory<float> matrix_C;
	matrix_C.host = (float*)aligned_malloc(alignedSize3, alignmentForPtr);
	if (!matrix_C.host)
	{
		cout << "out of memory\n";
		exit(0);
	}

	// Initialize matrices row by row.
	for (size_t i = 0; i < size1; ++i)
	{
		float* row_A = matrix_A.host + i*size3;
		// Fill the rows with random values from range [0, 1]
		fill_rand_uniform_01(row_A, size3);
	}

	// Initialize matrices row by row.
	for (size_t i = 0; i < size3; ++i)
	{
		float* row_C = matrix_C.host + i*size1;
		// Fill the rows with random values from range [0, 1]
		std::fill(row_C, row_C + size1, float(0));
	}

	cl_int err = 0; // OpenCL error code
	{
		matrix_A.device = clCreateBuffer(
			oclobjects.context,
			CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
			matrix1_memory_size,
			matrix_A.host,
			&err
			);
		SAMPLE_CHECK_ERRORS(err);
	}
	matrix_C.device = clCreateBuffer(
		oclobjects.context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		matrix3_memory_size,
		matrix_C.host,
		&err
		);
	SAMPLE_CHECK_ERRORS(err);
	//alpha and beta are not added into calculation; you could add them as you want...
	float alpha = 1;//rand_uniform_01<float>();
	float beta = 0;//rand_uniform_01<float>();
	//cout << "Using alpha = " << alpha << " and beta = " << beta << "\n";

	// -----------------------------------------------------------------------
	// Setting kernel arguments
	// -----------------------------------------------------------------------
	err = clSetKernelArg(executable.kernel, 0, sizeof(cl_mem), &matrix_A.device);
	SAMPLE_CHECK_ERRORS(err);
#ifdef USE_MATRIX_B
	err = clSetKernelArg(executable.kernel, 1, sizeof(cl_mem), &matrix_B.device);
	SAMPLE_CHECK_ERRORS(err);
#endif
	err = clSetKernelArg(executable.kernel, 1, sizeof(cl_mem), &matrix_C.device);
	SAMPLE_CHECK_ERRORS(err);
	err = clSetKernelArg(executable.kernel, 2, sizeof(int), &size1);
	SAMPLE_CHECK_ERRORS(err);
	err = clSetKernelArg(executable.kernel, 3, sizeof(int), &size3);
	SAMPLE_CHECK_ERRORS(err);

	//size_t global_size[2] = { size3 / block_x, size1 / block_y };
	size_t global_size[2] = { size3, size1};
	size_t local_size[2] = { block_y, block_x};// { block_y, block_y };
	/*
	cout << "Global size: "<<global_size[0] <<", "<< global_size[1] <<endl;
	cout << "Local size: " << local_size[0] << ", " << local_size[1] << endl;
	cout << "block_y size: " << block_y << ", " << block_x << endl;
	cout << "size1 size3: " << size1 << ", " << size3 << endl;
	*/

	// theoretical number of floating point operations (addition and multiplication) for one kernel execution
	// needed for performance calculations (GFLOPS) at every iteration below
	double flops = double(size1)*size3;
	//cout << "number of floating point operations: " << flops << endl;

	// -----------------------------------------------------------------------
	// Loop with the kernel invocation
	// -----------------------------------------------------------------------
	double total = 0.0;
	double total_kernel = 0.0;
	double minTime = 1e100;
	cl_event cl_perf_event = NULL;
	for (int i = 0; i < cmdparser.iterations.getValue(); ++i)
	{
		// Here we start measuring host time for kernel execution
		double start = time_stamp();

		err = clEnqueueNDRangeKernel(
			oclobjects.queue,
			executable.kernel,
			2,
			0,
			global_size,
			local_size,
			0, 0, &cl_perf_event
			);
		SAMPLE_CHECK_ERRORS(err);

		err = clWaitForEvents(1, &cl_perf_event);
		SAMPLE_CHECK_ERRORS(err);

		// It is important to measure end host time after clWaitForEvents call
		double end = time_stamp();

		double time = end - start;
		if (i > 0)
			total += time;
		
		//cout << endl;
		//cout << "Host time: " << time * 1e6 << " microsec.\n";
		//cout << "Host perf: " << flops / time / 1e9 << " GFLOPS\n";

		cl_ulong startk = 0;
		cl_ulong endk = 0;

		// notice that pure HW execution time is END-START
		err = clGetEventProfilingInfo(cl_perf_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startk, NULL);
		SAMPLE_CHECK_ERRORS(err);
		err = clGetEventProfilingInfo(cl_perf_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endk, NULL);
		SAMPLE_CHECK_ERRORS(err);
		float kernel_time = (float)(endk - startk)*1e-6f;

		if (i > 0)
		{
			if (kernel_time < minTime)
				minTime = kernel_time;
			total_kernel += kernel_time;
		}
		//cout << "Kernel time: " << kernel_time  * 1e6 << " microsec.\n";
		//cout << "Kernel perf: " << flops / kernel_time / 1e9 << " GFLOPS\n";

		if (i == 0 && cmdparser.validation.getValue())
		{
			// Validate result for the first iteration only and
			// only if user wants this.
			// Please note, validation procedure cannot be run at
			// futher iterations after the very first iteration,
			// as the results are being accumulated in C matrix
			// every iteration but validation procedures assumes that
			// C initial values are all zeros.

			clEnqueueMapBuffer(
				oclobjects.queue,
				matrix_C.device,
				CL_TRUE,    // blocking map
				CL_MAP_READ,
				0,
				matrix3_memory_size,
				0, 0, 0,
				&err
				);
			SAMPLE_CHECK_ERRORS(err);

			float *verificationOutput = (float*)malloc(alignedSize3);
			if (!verificationOutput)
			{
				cout << "out of memory\n";
				exit(0);
			}
			for (size_t i = 0; i < size1; ++i)
			{
				float* row_verificationOutput = verificationOutput + i*size3;
				std::fill(row_verificationOutput, row_verificationOutput + size3, float(0));
			}

			cpuTranspose(verificationOutput,
				matrix_A.host,
				size1,
				size3);
			//return;
			if (!check(size1, size3, matrix_C.host, verificationOutput))
			{
				throw Error("Validation procedure reported failures");
			}

			free(verificationOutput);

			//cout << "Verification passed\n";
			err = clEnqueueUnmapMemObject(
				oclobjects.queue,
				matrix_C.device,
				matrix_C.host,
				0, 0, 0
				);
			SAMPLE_CHECK_ERRORS(err);

			// Finish here is only required for correct time measurment on the next iteration
			// It does not affect correctness of calculations because you use the in-order OpenCL queue here.
			err = clFinish(oclobjects.queue);
			SAMPLE_CHECK_ERRORS(err);
		}
	}

	double avg = total / (cmdparser.iterations.getValue() - 1);
	double avg_kernel = total_kernel / (cmdparser.iterations.getValue() - 1);
	cout << setw(26) << kernel_name << "\t\t" << avg_kernel << "\t\t\t\t" <<  avg << "\t\t\t\t" <<  minTime<< endl;
	// All resources are deallocated automatically.
	/*
	if (cmdparser.iterations.getValue() == 1)
		cout << kernel_name << endl;
	else
		cout << kernel_name << "\t\t" << flops / avg_kernel / 1e9 <<"\t\t\t\t" << flops / avg / 1e9 << "\t\t\t\t" << flops / minTime / 1e9 << endl;
	*/
}

void run_kernel(CmdParserGEMM& cmdparser, OpenCLBasic& oclobjects, OpenGEMMProgram& executable, const string& kernel_name, int size1, int size3)
{
	cout << setw(26) << "Algorithm" << "\t"<<"Avg kernel ms"<<"\t\t\t\t"<<"Avg Host ms"<<"\t\t\t\t"<<"Peak Kernel ms" << endl;
	//naming convention: blockHeight x blockWidth_groupHeight x groupWidth
	if (kernel_name == "")
	{
		gemm_full(cmdparser, oclobjects, executable, "transpose_naive", size1, size3, 1, 1, 16, 16); //1, 16, 1, 1);
		gemm_full(cmdparser, oclobjects, executable, "transpose_coalesced", size1, size3, 1, 1, 16, 16); 
		gemm_full(cmdparser, oclobjects, executable, "transpose_nobancconflicts", size1, size3, 1, 1, 16, 16); 
	}
}

// Entry point for sample application, command-line parsing,
// generic OpenCL resources allocation and deallocation.
int main(int argc, const char** argv)
{
	try
	{
		// Define and parse command-line arguments.
		CmdParserGEMM cmdparser(argc, argv);
		cmdparser.parse();

		// Immediatly exit if user wanted to see the usage information only.
		if (cmdparser.help.isSet())
		{
			return 0;
		}

		// Create the necessary OpenCL objects up to device queue.
		OpenCLBasic oclobjects(
			cmdparser.platform.getValue(),
			cmdparser.device_type.getValue(),
			cmdparser.device.getValue(), CL_QUEUE_PROFILING_ENABLE
			);

		// Build Program
		string build_option = " -cl-mad-enable";
		if (cmdparser.device_type.getValue() == "gpu")
			build_option += " -DGPU";
		OpenGEMMProgram executable(oclobjects, L"transpose.cl", "", "", build_option);
		int M = cmdparser.size1.getValue();
		int K = cmdparser.size2.getValue();

		if (M != 0 && K != 0)
		{
			cout << "-----------------------------------------" << endl;
			cout << "matrix size:" << M << "x" << K << endl;
			run_kernel(cmdparser, oclobjects, executable, cmdparser.kernel.getValue(), M, K);
		}
#if 0
		else
		{
			cout << "\nM/K/N are not set, thus execute all the test from 512 * 512 * 512 to 2048 * 2048 * 2048" << endl;
			for (int size1 = 512; size1 <= 2048; size1 *= 2)
			{
				for (int size3 = 512; size3 <= 2048; size3 *= 2)
				{
					for (int size2 = 512; size2 <= 2048; size2 *= 2)
					{
						cout << "-----------------------------------------" << endl;
						cout << "matrix size: ( " << size1 << "x" << size2 << " ) * ( " << size2 << "x" << size3 << " )" << endl;
						run_kernel(cmdparser, oclobjects, executable, cmdparser.kernel.getValue(), size1, size2, size3);
					}
				}
			}
		}
#endif
		return 0;
	}
	catch (const CmdParser::Error& error)
	{
		cerr
			<< "[ ERROR ] In command line: " << error.what() << "\n"
			<< "Run " << argv[0] << " -h for usage info.\n";
		return 1;
	}
	catch (const Error& error)
	{
		cerr << "[ ERROR ] Sample application specific error: " << error.what() << "\n";
		return 1;
	}
	catch (const exception& error)
	{
		cerr << "[ ERROR ] " << error.what() << "\n";
		return 1;
	}
	catch (...)
	{
		cerr << "[ ERROR ] Unknown/internal error happened.\n";
		return 1;
	}
}
