#include "SVD.h"

using namespace cv;
using namespace lvi;
using namespace lvi::mat;
using namespace lvi::error;
using namespace lvi::edvr;

_declspec(dllexport) void CV_CORE_SVD_Create(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
    int32_t flags)
{
    handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc1, matInfoSrc1);

        EdvrObject<SVD> svdObj(edvrRef, new SVD(src, flags));

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_CORE_SVD_BackSubst(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    void* addressRhs, lvi::mat::MatInfo* matInfoRhs,
    void* addressDst, lvi::mat::MatInfo* matInfoDst)
{
    handleError(errorCluster, [&]() -> MgErr {
        Mat src = lvMatToCvMat(addressRhs, matInfoRhs);
        Mat dst = lvMatToCvMat(addressDst, matInfoDst);
        EdvrObject<SVD> svdObj(edvrRef);

		svdObj->backSubst(src, dst);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_CORE_SVD_Decompose(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
    int32_t flags)
{
    handleError(errorCluster, [&]() -> MgErr {
        Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
        EdvrObject<SVD> svdObj(edvrRef);

        (*svdObj)(src,flags);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_CORE_SVD_ReadU(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    void* addressDst, MatInfo* matInfoDst)
{
    handleError(errorCluster, [&]() -> MgErr {
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);
        EdvrObject<SVD> svdObj(edvrRef);
        
        svdObj->u.copyTo(dst);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_CORE_SVD_ReadUSize(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    int32_t* rows, int32_t* cols,
    uint32_t* type)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<SVD> svdObj(edvrRef);

        *rows = svdObj->u.rows;
        *cols = svdObj->u.cols;
        *type = static_cast<uint32_t>(svdObj->u.type() == CV_64F ? lvi::mat::DataType::Array2dDbl : lvi::mat::DataType::Array2dSgl);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_CORE_SVD_ReadVT(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    void* addressDst, MatInfo* matInfoDst)
{
    handleError(errorCluster, [&]() -> MgErr {
        Mat dst = lvMatToCvMat(addressDst, matInfoDst);
        EdvrObject<SVD> svdObj(edvrRef);

        svdObj->vt.copyTo(dst);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_CORE_SVD_ReadVTSize(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    int32_t* rows, int32_t* cols,
    uint32_t* type)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<SVD> svdObj(edvrRef);

        *rows = svdObj->vt.rows;
        *cols = svdObj->vt.cols;
        *type = static_cast<uint32_t>(svdObj->vt.type() == CV_64F ? lvi::mat::DataType::Array2dDbl : lvi::mat::DataType::Array2dSgl);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_CORE_SVD_ReadW(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    void* addressDst, MatInfo* matInfoDst)
{
    handleError(errorCluster, [&]() -> MgErr {
        Mat dst = lvMatToCvMat(addressDst, matInfoDst);
        EdvrObject<SVD> svdObj(edvrRef);

        svdObj->w.copyTo(dst);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_CORE_SVD_ReadWSize(
    lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    int32_t* rows, int32_t* cols,
    uint32_t* type)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<SVD> svdObj(edvrRef);

        *rows = svdObj->w.rows;
        *cols = svdObj->w.cols;
        *type = static_cast<uint32_t>(svdObj->w.type() == CV_64F ? lvi::mat::DataType::Array2dDbl : lvi::mat::DataType::Array2dSgl);

        return mgNoErr;
    });
}