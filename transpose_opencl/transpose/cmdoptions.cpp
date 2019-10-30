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


#include <limits>
#include <cmath>

#include "cmdoptions.hpp"

using namespace std;


#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4355)    // 'this': used in base member initializer list
#endif

CmdParserGEMM::CmdParserGEMM (int argc, const char** argv) :
    CmdParserCommon(argc, argv),
    size1(
        *this,
        'M',
        "size1",
        "<integer>",
        "rows of 1st matrix in elements.",
        0
    ),
	size2(
		*this,
		'K',
		"size2",
		"<integer>",
		"cols of 1st matrix in elements, rows of 2nd matrix in element",
		0
	),
	size3(
		*this,
		'N',
		"size3",
		"<integer>",
		"cols of 2nd matrix in elements",
		0
	),
    iterations(
        *this,
        'i',
        "iterations",
        "<integer>",
        "Number of kernel invocations. For each invoction, "
            "performance information will be printed. "
            "Zero is allowed: in this case no kernel invocation "
            " is performed but all other host stuff is created.",
        10
    ),
    kernel(
        *this,
        'k',
        "kernel",
        "",
        "Determines format of matrices involved in multiplication. "
            "There are several supported kernels with naive implementation and optimization on Intel GPU;"
            "both matrices A and B are in column-major form;",
        ""
    ),
	kernel_naive(kernel, "naive"),
	kernel_L3_SIMD_32x2_1x8(kernel, "L3_SIMD_32x2_1x8"),
	kernel_L3_SIMD_16x2_1x8(kernel, "L3_SIMD_16x2_1x8"),
	kernel_L3_SIMD_16x2_4x8(kernel, "L3_SIMD_16x2_4x8"),
	kernel_L3_SIMD_8x4_1x8(kernel, "L3_SIMD_8x4_1x8"),
	kernel_L3_SIMD_8x4_8x8(kernel, "L3_SIMD_8x4_8x8"),
	kernel_L3_SIMD_8x4_8x8_barrier(kernel, "L3_SIMD_8x4_8x8_barrier"),
	kernel_L3_SLM_8x8_8x16(kernel, "L3_SLM_8x8_8x16"),
	kernel_L3_SLM_8x8_4x16(kernel, "L3_SLM_8x8_4x16"),
	kernel_L3_SLM_8x8_16x16(kernel, "L3_SLM_8x8_16x16"),
	kernel_block_read_32x2_1x8(kernel, "block_read_32x2_1x8"),
	kernel_block_read_32x2_4x8(kernel, "block_read_32x2_4x8"),
	kernel_block_read_32x2_8x8(kernel, "block_read_32x2_8x8"),
	kernel_block_read_16x2_1x8(kernel, "block_read_16x2_1x8"),
	kernel_block_read_16x2_4x8(kernel, "block_read_16x2_4x8"),
	kernel_block_read_16x2_8x8(kernel, "block_read_16x2_8x8"),
	kernel_block_read_16x4_1x8(kernel, "block_read_16x4_1x8"),
	kernel_block_read_16x4_4x8(kernel, "block_read_16x4_4x8"),
	kernel_block_read_16x4_8x8(kernel, "block_read_16x4_8x8"),
	kernel_block_read_32x1_1x8(kernel, "block_read_32x1_1x8"),
	kernel_block_read_4x1_1x8(kernel, "block_read_4x1_1x8"),
    validation(
        *this,
        'v',
        "validation",
        "",
        "Enables validation procedure on host (slow for big matrices).",
        false
    )
{
}

#ifdef _MSC_VER
#pragma warning (pop)
#endif


void CmdParserGEMM::parse ()
{
    CmdParserCommon::parse();

    // Test a small part of parameters for consistency
    // in this function. The major part of checks is placed in
    // validateParameters function. But to call it you need
    // further specialization on what OpenCL objects and their
    // capabilities are.

    /*if(arithmetic_float.isSet() && arithmetic_double.isSet())
    {
        throw CmdParser::Error(
            "Both float and double are chosen. "
            "Should be only one of them."
        );
    }

    if(!arithmetic_float.isSet() && !arithmetic_double.isSet())
    {
        throw CmdParser::Error(
            "Neither float nor double are chosen. "
            "One of them should be chosen."
        );
    }*/
}


size_t CmdParserGEMM::estimateMaxMatrixSize (
    OpenCLBasic& oclobjects,
    size_t size_of_element,
    size_t alignment
)
{
    cl_ulong max_alloc_size = 0;
    cl_int err = clGetDeviceInfo(
        oclobjects.device,
        CL_DEVICE_MAX_MEM_ALLOC_SIZE,
        sizeof(max_alloc_size),
        &max_alloc_size,
        0
    );
    SAMPLE_CHECK_ERRORS(err);

    cl_ulong max_global_mem_size = 0;
    err = clGetDeviceInfo(
        oclobjects.device,
        CL_DEVICE_GLOBAL_MEM_SIZE,
        sizeof(max_global_mem_size),
        &max_global_mem_size,
        0
    );
    SAMPLE_CHECK_ERRORS(err);

    double max_matrix_size = sqrt(
        min(
            double(numeric_limits<size_t>::max()),
            min(double(max_alloc_size), double(max_global_mem_size)/3)
        ) / size_of_element
    );

    assert(alignment%size_of_element == 0);

    // the following is effect of a bit conservative
    // estimation of the overhead on a row alignment
    max_matrix_size -= alignment/size_of_element;

    assert(max_matrix_size < double(numeric_limits<size_t>::max()));

    return static_cast<size_t>(max_matrix_size);
}

void CmdParserGEMM::validateParameters (
    OpenCLBasic& oclobjects,
    OpenCLProgramOneKernel& executable,
    size_t size_of_element,
    size_t alignment
)
{
    validatePositiveness(size1);
	validatePositiveness(size2);
	validatePositiveness(size3);

	cl_ulong max_work_group_size = 0;
	int err = clGetDeviceInfo(
		oclobjects.device,
		CL_DEVICE_MAX_WORK_GROUP_SIZE,
		sizeof(max_work_group_size),
		&max_work_group_size,
		0
		);
	SAMPLE_CHECK_ERRORS(err);

    size_t max_matrix_size =
        estimateMaxMatrixSize(oclobjects, size_of_element, alignment);

    size1.validate(
        size1.getValue() <= max_matrix_size,
        "requested value is too big; should be <= " + to_str(max_matrix_size)
    );

	size2.validate(
		size2.getValue() <= max_matrix_size,
		"requested value is too big; should be <= " + to_str(max_matrix_size)
		);

	size3.validate(
		size3.getValue() <= max_matrix_size,
		"requested value is too big; should be <= " + to_str(max_matrix_size)
		);

    iterations.validate(
        iterations.getValue() >= 0,
        "negative value is provided; should be positive or zero"
    );
}
