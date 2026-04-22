#pragma once

#include "opencv2/dnn.hpp"
#include "../../../Interorp/Edvr/EdvrTypes.h"
#include "../../../Interorp/MatHandling/Mathandling.h"
#include "../../../Interorp/ErrorHandling/ErrorHandling.h"
#include "../../../Interorp/ArrayHandling/Array1D.hpp"
#include "../../../Interorp/ArrayHandling/Array4D.hpp"
#include "../../../Interorp/Points.hpp"

extern "C"
{

#pragma region Create

    _declspec(dllexport) void CV_DNN_BLOB_Create(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef); 

#pragma endregion Create

#pragma region BlobFromImage

    _declspec(dllexport) void CV_DNN_BLOB_BlobFromImage(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        void* addressImage, lvi::mat::MatInfo* matInfoImage,
        double scalefactor, lvi::point::PointI32* size,
        lvi::array::LV_1DArrayHandle<double_t> meanHandle,
        LVBoolean* swapRB, LVBoolean* crop, 
        int32_t ddepth);

    _declspec(dllexport) void CV_DNN_BLOB_BlobFromImages(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_1DArrayHandle<uint64_t> adressesHandle,
        lvi::array::LV_1DArrayHandle<lvi::mat::MatInfo> matInfoImagesHandle,
        double scalefactor, lvi::point::PointI32* size,
        lvi::array::LV_1DArrayHandle<double_t> meanHandle,
        LVBoolean* swapRB, LVBoolean* crop,
        int32_t ddepth);
#pragma endregion BlobFromImage

#pragma region Read


    //WIN32 packing
    #if defined(_WIN32) && !defined(_WIN64)
        #pragma pack(push, 1) 
        #define LV_PACKING_SET 1
    #endif

    struct BlobInfo
    {
        int32_t Dimensions;
        int32_t Channels;
        int32_t Height;
        int32_t Width;
        uint16_t Type;
    };

    #ifdef LV_PACKING_SET
        #pragma pack(pop)
        #undef LV_PACKING_SET
    #endif

    _declspec(dllexport) void CV_DNN_BLOB_ReadInfo(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        BlobInfo* blobInfo);

    _declspec(dllexport) void CV_DNN_BLOB_ReadData_U8(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_4DArrayHandle<uint8_t> nchwArrayHandle);

    _declspec(dllexport) void CV_DNN_BLOB_ReadData_SGL(lvi::error::ErrorClusterPtr errorCluster,
        lvi::edvr::ReferencePtr edvrRef,
        lvi::array::LV_4DArrayHandle<float> nchwArrayHandle);

#pragma endregion Read

}
