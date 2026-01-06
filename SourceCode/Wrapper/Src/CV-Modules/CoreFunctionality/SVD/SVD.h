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
    _declspec(dllexport) void CV_CORE_SVD_Create(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        int32_t flags);

    _declspec(dllexport) void CV_CORE_SVD_BackSubst(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressRhs, lvi::mat::MatInfo* matInfoRhs, 
        void* addressDst, lvi::mat::MatInfo* matInfoDst);

    _declspec(dllexport) void CV_CORE_SVD_Decompose(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        int32_t flags);

     _declspec(dllexport) void CV_CORE_SVD_ReadU(
        lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressDst, lvi::mat::MatInfo* matInfoDst);

     _declspec(dllexport) void CV_CORE_SVD_ReadUSize(
         lvi::error::ErrorClusterPtr errorCluster,
         lvi::edvr::ReferencePtr edvrRef,
         int32_t* rows, int32_t* cols,
         uint32_t* type);

     _declspec(dllexport) void CV_CORE_SVD_ReadVT(
         lvi::error::ErrorClusterPtr errorCluster,
         lvi::edvr::ReferencePtr edvrRef,
         void* addressDst, lvi::mat::MatInfo* matInfoDst);

     _declspec(dllexport) void CV_CORE_SVD_ReadVTSize(
         lvi::error::ErrorClusterPtr errorCluster,
         lvi::edvr::ReferencePtr edvrRef,
         int32_t* rows, int32_t* cols,
         uint32_t* type);

     _declspec(dllexport) void CV_CORE_SVD_ReadW(
         lvi::error::ErrorClusterPtr errorCluster,
         lvi::edvr::ReferencePtr edvrRef,
         void* addressDst, lvi::mat::MatInfo* matInfoDst);

     _declspec(dllexport) void CV_CORE_SVD_ReadWSize(
         lvi::error::ErrorClusterPtr errorCluster,
         lvi::edvr::ReferencePtr edvrRef,
         int32_t* rows, int32_t* cols,
         uint32_t* type);

}