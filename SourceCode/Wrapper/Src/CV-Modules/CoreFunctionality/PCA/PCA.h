#pragma once

#include "opencv2/core.hpp"
#include "../../../Interorp/StringHandling/StringHandling.h"
#include "../../../Interorp/Edvr/EdvrTypes.h"
#include "../../../Interorp/MatHandling/Mathandling.h"
#include "../../../Interorp/ErrorHandling/ErrorHandling.h"
#include "../../../Interorp/ArrayHandling/Array1D.hpp"
#include "../../../Interorp/ArrayHandling/Array2D.hpp"

using namespace std;
using namespace cv;

extern "C"
{
#pragma region Create 

    _declspec(dllexport) void CV_CORE_PCA_Create(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef);

    _declspec(dllexport) void CV_CORE_PCA_CreateMaxComponents(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        void* addressMean, lvi::mat::MatInfo* matInfoMean,
        int32_t flags, int32_t maxComponents);

    _declspec(dllexport) void CV_CORE_PCA_CreateRetainedVariance(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        void* addressMean, lvi::mat::MatInfo* matInfoMean,
        int32_t flags, double retainedVariance);

#pragma endregion Create 

#pragma region Compute

    _declspec(dllexport) void CV_CORE_PCA_ComputeMaxComponents(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        void* addressMean, lvi::mat::MatInfo* matInfoMean,
        int32_t flags, int32_t maxComponents);

    _declspec(dllexport) void CV_CORE_PCA_ComputeRetainedVariance(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        void* addressMean, lvi::mat::MatInfo* matInfoMean,
        int32_t flags, double retainedVariance);

#pragma endregion Compute 

#pragma region FileIO 

    _declspec(dllexport) void CV_CORE_PCA_Read(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* filePath);

    _declspec(dllexport) void CV_CORE_PCA_Write(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* filePath);

#pragma endregion FileIO

#pragma region Read 

    _declspec(dllexport) void CV_CORE_PCA_ReadEigenValues(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_1DArrayHandle<double_t> valuesHandle);

    _declspec(dllexport) void CV_CORE_PCA_ReadEigenVectors(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_2DArrayHandle<double_t> vectorsHandle);

    _declspec(dllexport) void CV_CORE_PCA_ReadMean(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_2DArrayHandle<double_t> meanHandle);

    _declspec(dllexport) void CV_CORE_PCA_ReadMeanSize(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        int32_t* rows, int32_t* cols);

    _declspec(dllexport) void CV_CORE_PCA_ReadEigenVectorsSize(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        int32_t* rows, int32_t* cols);

#pragma endregion Read 

#pragma region Project 

    _declspec(dllexport) void CV_CORE_PCA_BackProject(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        void* addressResult, lvi::mat::MatInfo* matInfoResult);

    _declspec(dllexport) void CV_CORE_PCA_Project(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        void* addressResult, lvi::mat::MatInfo* matInfoResult);

#pragma endregion Project

}