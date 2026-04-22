#pragma once

#include "opencv2/dnn.hpp"
#include "../../../Interorp/Edvr/EdvrTypes.h"
#include "../../../Interorp/MatHandling/Mathandling.h"
#include "../../../Interorp/ErrorHandling/ErrorHandling.h"
#include "../../../Interorp/ArrayHandling/Array1D.hpp"
#include "../../../Interorp/ArrayHandling/Array4D.hpp"
#include "../../../Interorp/StringHandling/StringHandling.h"
#include "../../../CV-Modules/DNN/DNN.h"

extern "C"
{

#pragma region Create

    _declspec(dllexport) void CV_DNN_NET_Create(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef);

    _declspec(dllexport) void CV_DNN_NET_ReadNet(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* modelPath, const char* configPath, const char* framework);

    _declspec(dllexport) void CV_DNN_NET_ReadNetFromONNX(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* modelPath);

    _declspec(dllexport) void CV_DNN_NET_ReadNetFromCaffe(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* prototxtPath, const char* caffeModelPath);

    _declspec(dllexport) void CV_DNN_NET_ReadNetFromTensorflow(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* modelPath, const char* configPath);

    _declspec(dllexport) void CV_DNN_NET_ReadNetFromTFLite(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* modelPath);

    _declspec(dllexport) void CV_DNN_NET_ReadNetFromTorch(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* modelPath, LVBoolean* isBinary,
        LVBoolean* evaluate);

    _declspec(dllexport) void CV_DNN_NET_ReadNetFromDarknet(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* cfgPath, const char* weightsPath);

    _declspec(dllexport) void CV_DNN_NET_ReadNetFromModelOptimizer(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* xml, const char* bin);

#pragma endregion Create

#pragma region Forward

    _declspec(dllexport) void CV_DNN_NET_Forward(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRefNet,
        const char* outputName,
        lvi::array::LV_4DArrayHandle<float> resultHandle);

    _declspec(dllexport) void CV_DNN_NET_ForwardMultiple(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRefNet,
        lvi::array::LV_1DArrayHandle<lvi::array::LV_4DArrayHandle<float>> resultHandle,
        lvi::array::LV_1DArrayHandle<LStrHandle> namesHandles);

    _declspec(dllexport) void CV_DNN_NET_SetInput(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRefNet,
        lvi::edvr::ReferencePtr edvrRefBlob,
        const char* name, double scalefactor,
        lvi::array::LV_1DArrayHandle<double> meanHandle);


#pragma endregion Forward

#pragma region Configuration

    _declspec(dllexport) void CV_DNN_NET_Connect(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* outPin, const char* inpPin);

    _declspec(dllexport) void CV_DNN_NET_SetPreferableBackend(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        int32_t backendId);

    _declspec(dllexport) void CV_DNN_NET_SetPreferableTarget(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        int32_t targetId);

    _declspec(dllexport) void CV_DNN_NET_EnableFusion(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        LVBoolean* fusion);

    _declspec(dllexport) void CV_DNN_NET_EnableWinograd(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        LVBoolean* useWinograd);

    _declspec(dllexport) void CV_DNN_NET_SetHalideScheduler(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* scheduler);

    _declspec(dllexport) void CV_DNN_NET_SetInputShape(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* inputName,
        lvi::array::LV_1DArrayHandle<int32_t> shapeHandle);

    _declspec(dllexport) void CV_DNN_NET_SetInputsNames(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::Array1D<LStrHandle> namesHandle);

    _declspec(dllexport) void CV_DNN_NET_RegisterOutput(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* outputName, int32_t layerId,
        int32_t outputPort, int32_t* boundLayer);

    _declspec(dllexport) void CV_DNN_NET_DumpToFile(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* path);

    _declspec(dllexport) void CV_DNN_NET_DumpToPbtxt(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* path);

#pragma endregion Configuration

#pragma region Layers

    _declspec(dllexport) void CV_DNN_NET_Empty(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        LVBoolean* result);

    _declspec(dllexport) void CV_DNN_NET_GetLayerId(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* layer,
        int32_t* layerId);

    _declspec(dllexport) void CV_DNN_NET_GetLayerNames(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::Array1D<LStrHandle> namesHandles);

    _declspec(dllexport) void CV_DNN_NET_GetLayersCount(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        const char* type,
        int32_t* count);

    _declspec(dllexport) void CV_DNN_NET_GetLayerTypes(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::Array1D<LStrHandle> typesHandles);

    _declspec(dllexport) void CV_DNN_NET_GetLayerShapes(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_1DArrayHandle<int32_t> netInputShapeHandle,
        int32_t layerId,
        lvi::array::LV_1DArrayHandle<lvi::array::LV_1DArrayHandle<int32_t>> inShapesHandle,
        lvi::array::LV_1DArrayHandle<lvi::array::LV_1DArrayHandle<int32_t>> outShapesHandle);

    _declspec(dllexport) void CV_DNN_NET_GetUnconnectedOutLayers(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_1DArrayHandle<int32_t> layersHandle);

    _declspec(dllexport) void CV_DNN_NET_GetUnconnectedOutLayersNames(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_1DArrayHandle<LStrHandle> namesHandles);

#pragma endregion Layers

#pragma region Performance   

    _declspec(dllexport) void CV_DNN_NET_GetFLOPS(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_1DArrayHandle<int32_t> netInputShapeHandle,
        int64_t* flops);

    _declspec(dllexport) void CV_DNN_NET_GetMemoryConsumption(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_1DArrayHandle<int32_t> netInputShapeHandle,
        int64_t* weights, int64_t* blobs);


    _declspec(dllexport) void CV_DNN_NET_GetPerfProfile(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_1DArrayHandle<double> timingsHandle,
        int64_t* total);

#pragma endregion Performance

}
