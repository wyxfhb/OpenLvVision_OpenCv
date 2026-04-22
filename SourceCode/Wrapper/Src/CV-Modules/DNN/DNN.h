#pragma once

#include "opencv2/dnn.hpp"
#include "../../Interorp/ErrorHandling/ErrorHandling.h"
#include "../../Interorp/MatHandling/Mathandling.h"
#include "../../Interorp/ArrayHandling/Array1D.hpp"
#include "../../Interorp/Points.hpp"

extern "C"
{

#pragma region Backends and Targets     



    struct BackendTarget
    {
        uint16_t backend;
        uint16_t target;
	};

	_declspec(dllexport) void CV_DNN_GetAvailableBackends(lvi::error::ErrorClusterPtr errorCluster,
        lvi::array::LV_1DArrayHandle<BackendTarget> backendArrayHandle);

    _declspec(dllexport) void CV_DNN_GetAvailableTargets(lvi::error::ErrorClusterPtr errorCluster,
        lvi::array::LV_1DArrayHandle<uint16_t> targetArrayHandle,
        int32_t backend);

#pragma endregion Backends and Targets  

#pragma region NMS

    #if defined(_WIN32) && !defined(_WIN64)
    #pragma pack(push, 1) 
    #define LV_PACKING_SET 1
    #endif

    struct LvRect2f
    {
        float x;
        float y;
        float width;
        float height;
    };
    
    struct LvRotatedRect
    {
        float cx;
        float cy;
        float width;
        float height;
        float angle;
    };


    #ifdef LV_PACKING_SET
    #pragma pack(pop)
    #undef LV_PACKING_SET
    #endif

    _declspec(dllexport) void CV_DNN_NMSBoxes(lvi::error::ErrorClusterPtr errorCluster,
        lvi::array::LV_1DArrayHandle<LvRect2f> bboxesHandle,
        lvi::array::LV_1DArrayHandle<float> scoresHandle,
        float scoreThreshold, float nmsThreshold,
        lvi::array::LV_1DArrayHandle<int32_t> indicesHandle,
        float eta, int32_t topK);

    _declspec(dllexport) void CV_DNN_NMSBoxesRotated(lvi::error::ErrorClusterPtr errorCluster,
        lvi::array::LV_1DArrayHandle<LvRotatedRect> bboxesHandle,
        lvi::array::LV_1DArrayHandle<float> scoresHandle,
        float scoreThreshold, float nmsThreshold,
        lvi::array::LV_1DArrayHandle<int32_t> indicesHandle,
        float eta, int32_t topK);

#pragma endregion NMS

}
