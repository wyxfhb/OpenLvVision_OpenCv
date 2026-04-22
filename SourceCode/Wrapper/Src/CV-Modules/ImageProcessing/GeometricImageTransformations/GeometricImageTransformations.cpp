#include "GeometricImageTransformations.h"

using namespace cv;
using namespace std;
using namespace lvi;
using namespace lvi::mat;
using namespace lvi::array;
using namespace lvi::point;
using namespace lvi::error;
using namespace lvi::edvr;

//CV_GEOMETRIC_ConvertMaps
//CV_GEOMETRIC_GetAffineTransform_Mat#
//CV_GEOMETRIC_GetAffineTransform_Points
//CV_GEOMETRIC_GetPerspectiveTransform
//CV_GEOMETRIC_GetRectSubPix

_declspec(dllexport) void CV_GEOMETRIC_GetRotationMatrix2D(lvi::error::ErrorClusterPtr errorCluster,
    lvi::point::PointSGL* center, double angle,
    double scale, lvi::array::LV_2DArrayHandle<double_t> resultHandle)
{
    handleError(errorCluster, [&]() -> MgErr {
		Array2D<double_t> resultMat(resultHandle);

	    Mat rotationMatrix = cv::getRotationMatrix2D(Point2f(center->x, center->y), angle, scale);
		return resultMat.copyFrom(rotationMatrix);
     });
}

//CV_GEOMETRIC_InvertAffineTransform
//CV_GEOMETRIC_LinearPolar
//CV_GEOMETRIC_LogPolar
//CV_GEOMETRIC_Remap
//CV_GEOMETRIC_Resize
//CV_GEOMETRIC_WarpAffine

_declspec(dllexport) void CV_GEOMETRIC_WarpAffine(lvi::error::ErrorClusterPtr errorCluster,
    void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
    void* addressDst, lvi::mat::MatInfo* matInfoDst,
    void* addressM, lvi::mat::MatInfo* matInfoM,
    lvi::point::PointI32* size,
    int32_t flags, int32_t borderMode,
    lvi::array::LV_1DArrayHandle<double_t> borderValueHandle)
{
    handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
        Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);
        Mat m = lvMatToCvMat(addressM, matInfoM);
		Array1D<double_t> borderValue(borderValueHandle);

        cv::warpAffine(src,dst,m,Size(size->x,size->y),flags,borderMode, borderValue.toScalar());

        return mgNoErr;
    });
}

//CV_GEOMETRIC_WarpPerspective
//CV_GEOMETRIC_WarpPolar
