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

#ifndef _INTEL_OPENCL_SAMPLE_GEMM_CMDOPTIONS_HPP_
#define _INTEL_OPENCL_SAMPLE_GEMM_CMDOPTIONS_HPP_

#include "oclobject.hpp"
#include "cmdparser.hpp"


// All command-line options for GEMM sample
class CmdParserGEMM : public CmdParserCommon
{
public:
    // For these options description, please refer to the constructor definition.

    CmdOption<int> size1;	//rows of 1st matrix in elements
	CmdOption<int> size2;	//cols of 1st matrix in elements, rows of 2nd matrix in element
	CmdOption<int> size3;	//cols of 2nd matrix in elements
    CmdOption<int> iterations;

    //CmdOption<string> arithmetic;
    //    CmdEnum<string> arithmetic_float;
    //    CmdEnum<string> arithmetic_double;

    CmdOption<string> kernel;
		CmdEnum<string> kernel_naive;
		CmdEnum<string> kernel_L3_SIMD_32x2_1x8;
		CmdEnum<string> kernel_L3_SIMD_16x2_1x8;
		CmdEnum<string> kernel_L3_SIMD_16x2_4x8;
		CmdEnum<string> kernel_L3_SIMD_8x4_1x8;
		CmdEnum<string> kernel_L3_SIMD_8x4_8x8;
		CmdEnum<string> kernel_L3_SIMD_8x4_8x8_barrier;
		CmdEnum<string> kernel_L3_SLM_8x8_8x16;
		CmdEnum<string> kernel_L3_SLM_8x8_4x16;
		CmdEnum<string> kernel_L3_SLM_8x8_16x16;
		CmdEnum<string> kernel_block_read_32x2_1x8;
		CmdEnum<string> kernel_block_read_32x2_4x8;
		CmdEnum<string> kernel_block_read_32x2_8x8;
		CmdEnum<string> kernel_block_read_16x2_1x8;
		CmdEnum<string> kernel_block_read_16x2_4x8;
		CmdEnum<string> kernel_block_read_16x2_8x8;
		CmdEnum<string> kernel_block_read_16x4_1x8;
		CmdEnum<string> kernel_block_read_16x4_4x8;
		CmdEnum<string> kernel_block_read_16x4_8x8;
		CmdEnum<string> kernel_block_read_32x1_1x8;
		CmdEnum<string> kernel_block_read_4x1_1x8;
    CmdOption<bool> validation;

    CmdParserGEMM (int argc, const char** argv);
    virtual void parse ();

    // Check if all parameters have correct and consistent
    // values based on device capabilities.
    void validateParameters (
        OpenCLBasic& oclobjects,
        OpenCLProgramOneKernel& executable,
        size_t size_of_element, // size of one element of matrix in bytes
        size_t alignment    // alignment requirements in bytes
    );

private:

    template <typename T>
    void validatePositiveness (const CmdOption<T>& parameter)
    {
        parameter.validate(
            parameter.getValue() > 0,
            "negative or zero value is provided; should be positive"
        );
    }

    size_t estimateMaxMatrixSize (
        OpenCLBasic& oclobjects,
        size_t size_of_element,
        size_t alignment
    );
};


#endif  // end of the include guard
