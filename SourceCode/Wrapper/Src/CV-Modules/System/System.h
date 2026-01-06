#pragma once

#include "opencv2/core.hpp"
#include "../../Interorp/ErrorHandling/ErrorHandling.h"
#include "../../Interorp/StringHandling/StringHandling.h"

using namespace std;
using namespace cv;

extern "C"
{
    _declspec(dllexport) void CV_SYSTEM_GetBuildInfo(
        lvi::error::ErrorClusterPtr errorCluster,
        LStrHandle hOutputString);

    _declspec(dllexport) void CV_SYSTEM_GetVersion(
        lvi::error::ErrorClusterPtr errorCluster,
        LStrHandle hOutputString);

    _declspec(dllexport) void CV_SYSTEM_SetNumThreads(
        lvi::error::ErrorClusterPtr errorCluster,
        int32_t nThreads);

    _declspec(dllexport) void CV_SYSTEM_GetNumThreads(
        lvi::error::ErrorClusterPtr errorCluster,
        int32_t* nThreads);

    _declspec(dllexport) void CV_SYSTEM_CheckHardwareSupport(
        lvi::error::ErrorClusterPtr errorCluster,
        int32_t featureId,
        LVBoolean* isSupported);
}