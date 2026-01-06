#pragma once
#include "extcode.h"
#include <cstdint>

namespace lvi
{
    namespace array
    {
        //WIN32 packing
        #if defined(_WIN32) && !defined(_WIN64)
            #pragma pack(push, 1) 
            #define LV_PACKING_SET 1
        #endif

        template <typename T>
        struct LV_1DArray
        {
            int32_t dimSize;
            T data[1];
        };
      
        template <typename T>
        struct LV_2DArray
        {
            int32_t dimSizes[2]; // [0] = rows, [1] = cols
            T data[1];
        };

        #ifdef LV_PACKING_SET
            #pragma pack(pop)
            #undef LV_PACKING_SET
        #endif
      
        template <typename T>
        using LV_1DArrayHandle = LV_1DArray<T>**;

        template <typename T>
        using LV_2DArrayHandle = LV_2DArray<T>**;

    } // namespace array
} // namespace lvi