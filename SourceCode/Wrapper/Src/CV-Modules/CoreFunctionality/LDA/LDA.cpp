#include "LDA.h"

using namespace cv;
using namespace lvi;
using namespace lvi::mat;
using namespace lvi::error;
using namespace lvi::edvr;

#pragma region Create 

_declspec(dllexport) void CV_CORE_LDA_CreateEmpty(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    int32_t numComponents)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::LDA> ldaObj(edvrRef, new cv::LDA(numComponents));

        return mgNoErr;
    });
}
 
_declspec(dllexport) void CV_CORE_LDA_CreateAndCompute(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    void* addressData, MatInfo* matInfoData,
    void* addressLabels, MatInfo* matInfoLabels,
    int32_t numComponents)
{
    handleError(errorCluster, [&]() -> MgErr {
        cv::Mat data = lvMatToCvMat(addressData, matInfoData);
        cv::Mat labels = lvMatToCvMat(addressLabels, matInfoLabels);
 
        EdvrObject<cv::LDA> ldaObj(edvrRef, new cv::LDA(data, labels, numComponents));       

        return mgNoErr; 
    });
}

#pragma endregion Create 

#pragma region Compute

_declspec(dllexport) void CV_CORE_LDA_Compute(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    void* addressData, MatInfo* matInfoData,
    void* addressLabels, MatInfo* matInfoLabels)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<LDA> ldaObj(edvrRef);    
        Mat data = lvMatToCvMat(addressData, matInfoData);
        Mat labels = lvMatToCvMat(addressLabels, matInfoLabels);

        ldaObj->compute(data, labels);

        return mgNoErr;
    });
}

#pragma endregion Compute

#pragma region FileIO 

_declspec(dllexport) void CV_CORE_LDA_Load(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* filePath)
{
    handleError(errorCluster, [&]() -> MgErr {

        EdvrObject<cv::LDA> ldaObj(edvrRef);
        ldaObj->load(filePath);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_CORE_LDA_Save(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* filePath)
{
    handleError(errorCluster, [&]() -> MgErr {

        EdvrObject<cv::LDA> ldaObj(edvrRef);
        ldaObj->save(filePath);

        return mgNoErr;
    });
}

#pragma endregion FileIO

#pragma region Read

_declspec(dllexport) void CV_CORE_LDA_ReadEigenValues(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    lvi::array::LV_1DArrayHandle<double_t> valuesHandle)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<LDA> ldaObj(edvrRef);

        Mat values = ldaObj->eigenvalues();

        lvi::array::Array1D<double_t> lvValues(valuesHandle);
        return lvValues.copyFrom(values);
    });
}

_declspec(dllexport) void CV_CORE_LDA_ReadEigenVectors(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    lvi::array::LV_2DArrayHandle<double_t> vectorsHandle)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::LDA> ldaObj(edvrRef);

        cv::Mat vectors = ldaObj->eigenvectors();

        lvi::array::Array2D<double_t> lvVectors(vectorsHandle);
        return lvVectors.copyFrom(vectors);
    });
}

_declspec(dllexport) void CV_CORE_LDA_ReadEigenVectorsSize(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    int32_t* rows, int32_t* cols) 
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::LDA> ldaObj(edvrRef);
        cv::Mat vectors = ldaObj->eigenvectors();

        *rows = vectors.rows;
        *cols = vectors.cols;

        return mgNoErr;
    });
}

#pragma endregion Read

#pragma region Reconstruct

_declspec(dllexport) void CV_CORE_LDA_Reconstruct(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    void* addressData, lvi::mat::MatInfo* matInfoData,
    lvi::array::LV_2DArrayHandle<double_t> dstHandle)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<LDA> ldaObj(edvrRef);
        Mat data = lvMatToCvMat(addressData, matInfoData);

        Mat result = ldaObj->reconstruct(data);

        lvi::array::Array2D<double_t> dst(dstHandle);
        return dst.copyFrom(result);
    });
}

#pragma endregion Reconstruct

#pragma region Project

_declspec(dllexport) void CV_CORE_LDA_Project(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    void* addressData, lvi::mat::MatInfo* matInfoData,
    lvi::array::LV_2DArrayHandle<double_t> dstHandle)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<LDA> ldaObj(edvrRef); 
        Mat data = lvMatToCvMat(addressData, matInfoData);

        Mat result = ldaObj->project(data);

        lvi::array::Array2D<double_t> dst(dstHandle);
        return dst.copyFrom(result);
    });
}
#pragma endregion Project