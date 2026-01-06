#pragma once

#include "opencv2/core.hpp"
#include "../../../Interorp/MatHandling/Mathandling.h"
#include "../../../Interorp/ErrorHandling/ErrorHandling.h"
#include "../../../Interorp/MatHandling/Mathandling.h"

using namespace std;
using namespace cv;

extern "C"
{
    _declspec(dllexport) void CV_CORE_RNG_Create(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef, uint64 state);

    _declspec(dllexport) void CV_CORE_RNG_Fill(lvi::error::ErrorClusterPtr errorCluster,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressA, lvi::mat::MatInfo* matInfoA,
        void* addressB, lvi::mat::MatInfo* matInfoB,
        lvi::edvr::ReferencePtr edvrRef, int distType,
		LVBoolean* saturate);

    _declspec(dllexport) void CV_CORE_RNG_Gaussian(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef, double* number, double sigma);

    _declspec(dllexport) void CV_CORE_RNG_Next(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef, uint32_t* number);

    //CV_CORE_RNG_Opperator
 
    _declspec(dllexport) void CV_CORE_RNG_ReadState(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef, uint64_t* state);

    _declspec(dllexport) void CV_CORE_RNG_Uniform_Int(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef, int32_t* number,
        int32_t a, int32_t b);

    _declspec(dllexport) void CV_CORE_RNG_Uniform_Float(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef, float32* number,
        float32 a, float32 b);

    _declspec(dllexport) void CV_CORE_RNG_Uniform_Double(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef, float64* number,
        float64 a, float64 b);

    _declspec(dllexport) void CV_CORE_RNG_WriteState(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef, uint64_t state);

}