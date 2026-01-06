#pragma once

#include "opencv2/core.hpp"
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

    _declspec(dllexport) void CV_CORE_LDA_CreateEmpty(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        int32_t numComponents);

    _declspec(dllexport) void CV_CORE_LDA_CreateAndCompute(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        void* addressLabels, lvi::mat::MatInfo* matInfoLabels,
        int32_t numComponents);

#pragma endregion Create 

#pragma region Compute

    _declspec(dllexport) void CV_CORE_LDA_Compute(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressData, lvi::mat::MatInfo* matInfoData,
		void* addressLabels, lvi::mat::MatInfo* matInfoLabels);

#pragma endregion Compute

#pragma region FileIO 

    _declspec(dllexport) void CV_CORE_LDA_Load(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* filePath);

	_declspec(dllexport) void CV_CORE_LDA_Save(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* filePath);

#pragma endregion FileIO

#pragma region Read

    _declspec(dllexport) void CV_CORE_LDA_ReadEigenValues(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_1DArrayHandle<double_t> valuesHandle);
    _declspec(dllexport) void CV_CORE_LDA_ReadEigenVectors(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_2DArrayHandle<double_t> vectorsHandle);

    _declspec(dllexport) void CV_CORE_LDA_ReadEigenVectorsSize(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        int32_t* rows, int32_t* cols);

#pragma endregion Read

#pragma region Reconstruct

    _declspec(dllexport) void CV_CORE_LDA_Reconstruct(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        lvi::array::LV_2DArrayHandle<double_t> dstHandle);

#pragma endregion Reconstruct

#pragma region Project

    _declspec(dllexport) void CV_CORE_LDA_Project(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        lvi::array::LV_2DArrayHandle<double_t> dstHandle);

#pragma endregion Project

}