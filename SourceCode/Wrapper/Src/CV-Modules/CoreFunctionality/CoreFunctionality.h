#pragma once

#include "opencv2/core.hpp"
#include "../../Interorp/MatHandling/Mathandling.h"
#include "../../Interorp/ArrayHandling/Array1D.hpp"
#include "../../Interorp/Points.hpp"
#include "../../Interorp/ErrorHandling/ErrorHandling.h"
#include "../../Interorp/Edvr/EdvrTypes.h"
#include "../../Interorp/ArrayHandling/Array2D.hpp"

extern "C"
{
#pragma region ArithmeticOperations

    _declspec(dllexport) void CV_CORE_AbsDiff(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        lvi::array::LV_1DArrayHandle<double_t> constantHandle);

    _declspec(dllexport) void CV_CORE_Add(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        lvi::array::LV_1DArrayHandle<double_t> constantHandle,
        void* addressMask, lvi::mat::MatInfo* matInfoMask);

    _declspec(dllexport) void CV_CORE_AddWeighted(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        double alpha, double beta, double gamma);

    _declspec(dllexport) void CV_CORE_ConvertScaleAbs(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        double alpha, double beta);

    _declspec(dllexport) void CV_CORE_Divide(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        lvi::array::LV_1DArrayHandle<double_t> constantHandle, double scale);

    _declspec(dllexport) void CV_CORE_Exp(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst);

    _declspec(dllexport) void CV_CORE_Log(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst);

    _declspec(dllexport) void CV_CORE_Multiply(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        lvi::array::LV_1DArrayHandle<double_t> constantHandle, double scale);

    _declspec(dllexport) void CV_CORE_Pow(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        double power);

    _declspec(dllexport) void CV_CORE_ScaleAdd(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        double alpha);

    _declspec(dllexport) void CV_CORE_Sqrt(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst);

    _declspec(dllexport) void CV_CORE_Subtract(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        lvi::array::LV_1DArrayHandle<double_t> constantHandle,
        void* addressMask, lvi::mat::MatInfo* matInfoMask);

    _declspec(dllexport) void CV_CORE_Sum(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        lvi::array::LV_1DArrayHandle<double_t> valuesHandle);

#pragma endregion ArithmeticOperations

#pragma region BitwiseOperations

    _declspec(dllexport) void CV_CORE_BitwiseAnd(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        void* addressMask, lvi::mat::MatInfo* matInfoMask);

    _declspec(dllexport) void CV_CORE_BitwiseNot(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressMask, lvi::mat::MatInfo* matInfoMask);

    _declspec(dllexport) void CV_CORE_BitwiseOr(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        void* addressMask, lvi::mat::MatInfo* matInfoMask);

    _declspec(dllexport) void CV_CORE_BitwiseXor(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        void* addressMask, lvi::mat::MatInfo* matInfoMask);

#pragma endregion BitwiseOperations

#pragma region ChannelManipulation

    _declspec(dllexport) void CV_CORE_ExtractChannel(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t coi);

    _declspec(dllexport) void CV_CORE_InsertChannel(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t coi);

    _declspec(dllexport) void CV_CORE_LUT(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressLut, lvi::mat::MatInfo* matInfoLut);

    _declspec(dllexport) void CV_CORE_Merge(lvi::error::ErrorClusterPtr errorCluster,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressCH0, lvi::mat::MatInfo* matInfoCH0,
        void* addressCH1, lvi::mat::MatInfo* matInfoCH1,
        void* addressCH2, lvi::mat::MatInfo* matInfoCH2);

    _declspec(dllexport) void CV_CORE_MixChannels(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int* fromTo, uint32_t size, LVBoolean* zeroChannels);

    _declspec(dllexport) void CV_CORE_Split(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressCH0, lvi::mat::MatInfo* matInfoCH0,
        void* addressCH1, lvi::mat::MatInfo* matInfoCH1,
        void* addressCH2, lvi::mat::MatInfo* matInfoCH2);

#pragma endregion ChannelManipulation

#pragma region Comparison

    _declspec(dllexport) void CV_CORE_CheckRange(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        LVBoolean* quiet,
        lvi::point::PointI32* pos,
        double minVal, double maxVal, LVBoolean* result);

    _declspec(dllexport) void CV_CORE_Compare(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        lvi::array::LV_1DArrayHandle<double_t> constantHandle,
        cv::CmpTypes cmpType);

    _declspec(dllexport) void CV_CORE_InRange(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressLowerb, lvi::mat::MatInfo* matLowerb,
        void* addressUpperb, lvi::mat::MatInfo* matUpperb,
        void* addressDst, lvi::mat::MatInfo* matInfoDst);

#pragma endregion Comparison

#pragma region DataConcatenation
    _declspec(dllexport) void CV_CORE_Hconcat(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2);

    _declspec(dllexport) void CV_CORE_Vconcat(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2);
#pragma endregion DataConcatenation

#pragma region FourierTransforms
    _declspec(dllexport) void CV_CORE_Dct(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        LVBoolean* inverse,
        LVBoolean* rows);

    _declspec(dllexport) void CV_CORE_Dft_Real(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        LVBoolean* rows, LVBoolean* scale);

    _declspec(dllexport) void CV_CORE_Dft_Complex(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        LVBoolean* rows, LVBoolean* scale);

    _declspec(dllexport) void CV_CORE_GetOptimalDftSize(lvi::error::ErrorClusterPtr errorCluster,
        int32_t vecsize,
        int32_t* optimalSizeOut);

    _declspec(dllexport) void CV_CORE_Idct(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        LVBoolean* rows);

    _declspec(dllexport) void CV_CORE_IDft_Real(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        LVBoolean* rows, LVBoolean* scale);

    _declspec(dllexport) void CV_CORE_IDft_Complex(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        LVBoolean* rows, LVBoolean* scale);

    _declspec(dllexport) void CV_CORE_MulSpectrums(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrcA, lvi::mat::MatInfo* matInfoSrcA,
        void* addressSrcB, lvi::mat::MatInfo* matInfoSrcB,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        LVBoolean* rows, LVBoolean* conjB);
#pragma endregion FourierTransforms

#pragma region GeometricOperations
    _declspec(dllexport) void CV_CORE_CartToPolar(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressArray1, lvi::mat::MatInfo* matInfoArray1,
        void* addressArray2, lvi::mat::MatInfo* matInfoArray2,
        LVBoolean* angleInDegrees);

    _declspec(dllexport) void CV_CORE_Magnitude(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressArray1, lvi::mat::MatInfo* matInfoArray1,
        void* addressArray2, lvi::mat::MatInfo* matInfoArray2);
        
    _declspec(dllexport) void CV_CORE_Phase(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressArray1, lvi::mat::MatInfo* matInfoArray1,
        void* addressArray2, lvi::mat::MatInfo* matInfoArray2,
        LVBoolean* angleInDegrees);

    _declspec(dllexport) void CV_CORE_PolarToCart(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressArray1, lvi::mat::MatInfo* matInfoArray1,
        void* addressArray2, lvi::mat::MatInfo* matInfoArray2,
        LVBoolean* angleInDegrees);

#pragma endregion GeometricOperations

#pragma region LDA

    _declspec(dllexport) void CV_CORE_LDA_SubspaceProject(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressW, lvi::mat::MatInfo* matInfoW,
        void* addressMean, lvi::mat::MatInfo* matInfoMean,
        void* addressResult, lvi::mat::MatInfo* matInfoResult);

    _declspec(dllexport) void CV_CORE_LDA_SubspaceReconstruct(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressProjected, lvi::mat::MatInfo* matInfoProjected,
        void* addressW, lvi::mat::MatInfo* matInfoW,
        void* addressMean, lvi::mat::MatInfo* matInfoMean,
        void* addressResult, lvi::mat::MatInfo* matInfoResult);


#pragma endregion LDA

#pragma region LinearAlgebra
    _declspec(dllexport) void CV_CORE_CompleteSymm(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,        
        LVBoolean* lowerToUpper);

    _declspec(dllexport) void CV_CORE_Determinant(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        double* determinant);

    _declspec(dllexport) void CV_CORE_Eigen(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        lvi::array::LV_1DArrayHandle<double_t> eigenValuesHandle,
        void* addressEigenVectors, lvi::mat::MatInfo* matEigenVectors);

    _declspec(dllexport) void CV_CORE_EigenNonSymmetric(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        lvi::array::LV_1DArrayHandle<double_t> eigenValuesHandle,
        void* addressEigenVectors, lvi::mat::MatInfo* matEigenVectors);

    _declspec(dllexport) void CV_CORE_Gemm(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,        
        void* addressSrc3, lvi::mat::MatInfo* matInfoSrc3,    
        double alpha,
        double beta,
        int flags);

    _declspec(dllexport) void CV_CORE_Invert(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        double* value, int flags);
    
    _declspec(dllexport) void CV_CORE_MulTransposed(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressDelta, lvi::mat::MatInfo* matInfoDelta,
        LVBoolean* aTa, double scale);

    _declspec(dllexport) void CV_CORE_Norm(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        void* addressMask, lvi::mat::MatInfo* matInfoMask,
		double* normValue, int normType);

    _declspec(dllexport) void CV_CORE_PatchNaNs(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressA, lvi::mat::MatInfo* matInfoA,
        double value);

    _declspec(dllexport) void CV_CORE_SetIdentity(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressMtx, lvi::mat::MatInfo* matInfoMtx,
        lvi::array::LV_1DArrayHandle<double_t> sHandle);

    _declspec(dllexport) void CV_CORE_Solve(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        lvi::array::LV_2DArrayHandle<double_t> dstHandle,
        int32_t flags,
        LVBoolean* solutionFound);

    _declspec(dllexport) void CV_CORE_SolveCubic(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressCoeffs, lvi::mat::MatInfo* matInfoCoeffs,
        lvi::array::LV_1DArrayHandle<double> rootsHandle);

    _declspec(dllexport) void CV_CORE_SolvePoly(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressCoeffs, lvi::mat::MatInfo* matInfoCoeffs,
        lvi::array::LV_1DArrayHandle<double_t> rootsHandle,
        int32_t maxIter);    

#pragma endregion LinearAlgebra

#pragma region MatrixImageTransformations
    _declspec(dllexport) void CV_CORE_BorderInterpolate(
        lvi::error::ErrorClusterPtr errorCluster,
        int32_t p, int32_t len,
        int32_t borderType, int32_t* value);

    _declspec(dllexport) void CV_CORE_CopyMakeBorder(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t top, int32_t bottom, int32_t left, int32_t right,
        int32_t borderType,
        lvi::array::LV_1DArrayHandle<double_t> valueHandle);

    _declspec(dllexport) void CV_CORE_Flip(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t FlipCode);

    _declspec(dllexport) void CV_CORE_FlipNd(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t axis);

    _declspec(dllexport) void CV_CORE_PerspectiveTransform(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressM, lvi::mat::MatInfo* matInfoM);

    _declspec(dllexport) void CV_CORE_Repeat(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t ny, int32_t nx);

    _declspec(dllexport) void CV_CORE_Rotate(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t rotateCode);

    _declspec(dllexport) void CV_CORE_TransformTo2D(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressM, lvi::mat::MatInfo* matInfoM);

    _declspec(dllexport) void CV_CORE_TransformTo3D(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressM, lvi::mat::MatInfo* matInfoM);

    _declspec(dllexport) void CV_CORE_TransformTo4D(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressM, lvi::mat::MatInfo* matInfoM);
    
    _declspec(dllexport) void CV_CORE_Transpose(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst);


#pragma endregion MatrixImageTransformations

#pragma region PCA

    _declspec(dllexport) void CV_CORE_PCACompute(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        lvi::array::LV_2DArrayHandle<double_t> meanHandle,
        lvi::array::LV_2DArrayHandle<double_t> eigenvectorsHandle,
        lvi::array::LV_1DArrayHandle<double_t> eigenvaluesHandle,
        int32_t maxComponents);

    _declspec(dllexport) void CV_CORE_PCABackProject(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        void* addressMean, lvi::mat::MatInfo* matInfoMean,
        void* addressEigVec, lvi::mat::MatInfo* matInfoEigVec,
        void* addressResult, lvi::mat::MatInfo* matInfoResult);

    _declspec(dllexport) void CV_CORE_PCAProject(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressData, lvi::mat::MatInfo* matInfoData,
        void* addressMean, lvi::mat::MatInfo* matInfoMean,
        void* addressEigVec, lvi::mat::MatInfo* matInfoEigVec,
        void* addressResult, lvi::mat::MatInfo* matInfoResult);

#pragma endregion 

#pragma region RandomNumber

    _declspec(dllexport) void CV_CORE_Randn(lvi::error::ErrorClusterPtr errorCluster,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressMean, lvi::mat::MatInfo* matInfoMean,
        void* addressStddev, lvi::mat::MatInfo* matInfoStddev);

    _declspec(dllexport) void CV_CORE_RandShuffle(lvi::error::ErrorClusterPtr errorCluster,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        lvi::edvr::ReferencePtr edvrRef, double iterFactor);

    _declspec(dllexport) void CV_CORE_Randu(lvi::error::ErrorClusterPtr errorCluster,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressLow, lvi::mat::MatInfo* matInfoLow,
        void* addressHigh, lvi::mat::MatInfo* matInfoHigh);

    _declspec(dllexport) void CV_CORE_WriteRngSeed(lvi::error::ErrorClusterPtr errorCluster,
        int32 seed);

#pragma endregion 

#pragma region Reduction

    _declspec(dllexport) void CV_CORE_Reduce(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t dim, cv::ReduceTypes rtype);

    _declspec(dllexport) void CV_CORE_ReduceArgMax(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t axis, LVBoolean* lastIndex);

    _declspec(dllexport) void CV_CORE_ReduceArgMin(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int32_t dim, LVBoolean* lastIndex);

    _declspec(dllexport) void CV_CORE_Trace(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        lvi::array::LV_1DArrayHandle<double_t> valuesHandle);

#pragma endregion

#pragma region SimilarityMetrics

    _declspec(dllexport) void CV_CORE_BatchDistance(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        void* addressMask, lvi::mat::MatInfo* matInfoMask,
        lvi::array::LV_2DArrayHandle<float> distHandle,
        lvi::array::LV_2DArrayHandle<int32_t> nidxHandle,
        int32_t normType, int32_t K,
        int32_t update, LVBoolean* crosscheck);


    _declspec(dllexport) void CV_CORE_Mahalanobis(lvi::error::ErrorClusterPtr errorCluster,
        void* addressV1, lvi::mat::MatInfo* matInfoV1,
        void* addressV2, lvi::mat::MatInfo* matInfoV2,
        void* addressIcovar, lvi::mat::MatInfo* matInfoIcovar,
        double_t * value);
    
    _declspec(dllexport) void CV_CORE_PSNR(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        double_t* value, double_t R);


#pragma endregion

#pragma region Sort

    _declspec(dllexport) void CV_CORE_Sort(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int sortFlags);

    _declspec(dllexport) void CV_CORE_SortIdx(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        int sortFlags);

#pragma endregion

#pragma region Statistical
    _declspec(dllexport) void CV_CORE_CountNonZero(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        int32_t* count);

    _declspec(dllexport) void CV_CORE_FindNonZero(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        lvi::array::LV_1DArrayHandle<lvi::point::PointI32> locationsHandle);

    _declspec(dllexport) void CV_CORE_HasNonZero(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        LVBoolean* value);

    _declspec(dllexport) void CV_CORE_Max(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        double value);

    _declspec(dllexport) void CV_CORE_Mean(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressMask, lvi::mat::MatInfo* matInfoMask,
        lvi::array::LV_1DArrayHandle<double_t> meanHandle);

    _declspec(dllexport) void CV_CORE_MeanStdDev(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressMask, lvi::mat::MatInfo* matInfoMask,
        lvi::array::LV_1DArrayHandle<double_t> meanHandle,
        lvi::array::LV_1DArrayHandle<double_t> stddevHandle);

    _declspec(dllexport) void CV_CORE_Min(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
        double value);

    _declspec(dllexport) void CV_CORE_MinMaxIdx(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressMask, lvi::mat::MatInfo* matInfoMask,
        double* minVal, double* maxVal, int* minIdx, int* maxIdx);

    _declspec(dllexport) void CV_CORE_MinMaxLoc(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressMask, lvi::mat::MatInfo* matInfoMask,
        double* minVal, double* maxVal,
        lvi::point::PointI32* minLoc,
        lvi::point::PointI32* maxLoc);

    _declspec(dllexport) void CV_CORE_Normalize(lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
        void* addressDst, lvi::mat::MatInfo* matInfoDst,
        void* addressMask, lvi::mat::MatInfo* matInfoMask,
        double alpha, double beta, cv::NormTypes normType);

#pragma endregion

#pragma region SVD

    _declspec(dllexport) void CV_CORE_SVBackSubst(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressW, lvi::mat::MatInfo* infoW,
        void* addressU, lvi::mat::MatInfo* infoU,
        void* addressVt, lvi::mat::MatInfo* infoVt,
        void* addressRhs, lvi::mat::MatInfo* infoRhs,
        void* addressDst, lvi::mat::MatInfo* infoDst);

    _declspec(dllexport) void CV_CORE_SVDecomp(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* infoSrc,
        void* addressW, lvi::mat::MatInfo* infoW,
        void* addressU, lvi::mat::MatInfo* infoU,
        void* addressVt, lvi::mat::MatInfo* infoVt,
        int32_t flags);

    _declspec(dllexport) void CV_CORE_SVD_Compute(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* infoSrc,
        void* addressW, lvi::mat::MatInfo* infoW,
        int32_t flags);

    _declspec(dllexport) void CV_CORE_SVD_SolveZ(
        lvi::error::ErrorClusterPtr errorCluster,
        void* addressSrc, lvi::mat::MatInfo* infoSrc,
        void* addressDst, lvi::mat::MatInfo* infoDst);

#pragma endregion

}