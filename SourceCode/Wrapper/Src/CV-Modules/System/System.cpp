#include "System.h"

using namespace cv;
using namespace std;
using namespace lvi;
using namespace lvi::error;
using namespace lvi::string;


_declspec(dllexport) void CV_SYSTEM_GetBuildInfo(
    lvi::error::ErrorClusterPtr errorCluster,
    LStrHandle hOutputString)
{
    handleError(errorCluster, [&]() -> MgErr {

        std::string ver = cv::getBuildInformation();
        setLvStringHandle(hOutputString, ver);

        return mgNoErr;
    });
}


_declspec(dllexport) void CV_SYSTEM_GetVersion(
    lvi::error::ErrorClusterPtr errorCluster,
    LStrHandle hOutputString)
{
    handleError(errorCluster, [&]() -> MgErr {

        std::string ver = cv::getVersionString();
        setLvStringHandle(hOutputString, ver);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_SYSTEM_SetNumThreads(
    lvi::error::ErrorClusterPtr errorCluster,
    int32_t nThreads)
{
    handleError(errorCluster, [&]() -> MgErr {

        cv::setNumThreads(nThreads);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_SYSTEM_GetNumThreads(
    lvi::error::ErrorClusterPtr errorCluster,
    int32_t* nThreads)
{
    handleError(errorCluster, [&]() -> MgErr {

        *nThreads = cv::getNumThreads();

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_SYSTEM_CheckHardwareSupport(
    lvi::error::ErrorClusterPtr errorCluster,
    int32_t featureId,
    LVBoolean* isSupported)
{
    handleError(errorCluster, [&]() -> MgErr {

        bool support = cv::checkHardwareSupport(featureId);
        *isSupported = support ? LVBooleanTrue : LVBooleanFalse;

        return mgNoErr;
    });
}