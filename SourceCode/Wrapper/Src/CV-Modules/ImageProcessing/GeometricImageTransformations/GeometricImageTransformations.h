#pragma once

#include "opencv2/imgproc.hpp"
#include "../../../Interorp/MatHandling/Mathandling.h"
#include "../../../Interorp/ErrorHandling/ErrorHandling.h"
#include "../../../Interorp/ArrayHandling/Array1D.hpp"
#include "../../../Interorp/ArrayHandling/Array2D.hpp"
#include "../../../Interorp/Points.hpp"

using namespace std;
using namespace cv;

extern "C" {

    // --- Maps & Remapping ---
    _declspec(dllexport) void CV_GEOMETRIC_ConvertMaps(lvi::error::ErrorClusterPtr errorCluster,
        void* addressMap1, lvi::mat::MatInfo* matInfoMap1,
        void* addressMap2, lvi::mat::MatInfo* matInfoMap2,
        void* addressDstMap1, lvi::mat::MatInfo* matInfoDstMap1,
        void* addressDstMap2, lvi::mat::MatInfo* matInfoDstMap2,
        int32_t dstmap1type, LVBoolean* nninterpolation);

    _declspec(dllexport) void CV_GEOMETRIC_GetAffineTransform_Mat(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrcPoints, int32_t srcCount,
        void* addressDstPoints, int32_t dstCount,
        lvi::array::LV_2DArrayHandle<double_t> result);

    _declspec(dllexport) void CV_GEOMETRIC_GetAffineTransform_Points(lvi::error::ErrorClusterPtr errorCluster,
        lvi::array::LV_1DArrayHandle<lvi::point::PointSGL> scr,
        lvi::array::LV_1DArrayHandle<lvi::point::PointSGL> dst,
        lvi::array::LV_2DArrayHandle<double_t> result);

    _declspec(dllexport) void CV_GEOMETRIC_GetPerspectiveTransform(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrcPoints, int32_t srcCount,
        void* addressDstPoints, int32_t dstCount,
        int32_t solveMethod,
        lvi::array::LV_2DArrayHandle<double_t> result);

    _declspec(dllexport) void CV_GEOMETRIC_GetRectSubPix(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t patchWidth, int32_t patchHeight,
        lvi::point::PointSGL* center, int32_t patchType);

    _declspec(dllexport) void CV_GEOMETRIC_GetRotationMatrix2D(lvi::error::ErrorClusterPtr errorCluster,
        lvi::point::PointSGL* center, double angle,
        double scale, lvi::array::LV_2DArrayHandle<double_t> resultHandle);

    _declspec(dllexport) void CV_GEOMETRIC_InvertAffineTransform(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrcMat, lvi::mat::MatInfo* matInfoSrc,
        void* addressDstMat, lvi::mat::MatInfo* matInfoDst);

    _declspec(dllexport) void CV_GEOMETRIC_LinearPolar(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        lvi::point::PointSGL* center, double maxRadius,
        int32_t flags);

    _declspec(dllexport) void CV_GEOMETRIC_LogPolar(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        lvi::point::PointSGL center, double M,
        int32_t flags);

    _declspec(dllexport) void CV_GEOMETRIC_Remap(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressMap1, lvi::mat::MatInfo* matInfoMap1,
        void* addressMap2, lvi::mat::MatInfo* matInfoMap2,
        int32_t interpolation, int32_t borderMode,
        lvi::array::LV_1DArrayHandle<double_t> borderValue);

    _declspec(dllexport) void CV_GEOMETRIC_Resize(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t dWidth, int32_t dHeight,
        double fx, double fy,
        int32_t interpolation);

    _declspec(dllexport) void CV_GEOMETRIC_WarpAffine(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressM, lvi::mat::MatInfo* matInfoM,
        lvi::point::PointI32* size,
        int32_t flags, int32_t borderMode,
        lvi::array::LV_1DArrayHandle<double_t> borderValueHandle);

    _declspec(dllexport) void CV_GEOMETRIC_WarpPerspective(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressM, lvi::mat::MatInfo* matInfoM,
        lvi::point::PointI32* size,
        int32_t flags, int32_t borderMode,
        lvi::array::LV_1DArrayHandle<double_t> borderValue);

    _declspec(dllexport) void CV_GEOMETRIC_WarpPolar(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t dWidth, int32_t dHeight,
        lvi::point::PointSGL center, double maxRadius,
        int32_t flags);
}