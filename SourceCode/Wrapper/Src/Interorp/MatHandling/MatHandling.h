#pragma once
#include "opencv2/opencv.hpp"
#include <functional>

/**
 * @brief Namespace LabVIEW Interorp
 */
namespace lvi
{
    /**
     * @brief Namespace for MatHandling 
     */
    namespace mat
    {
        /**
         * @brief Enum for LabVIEW data types passed to C++.
         */
        enum class DataType : uint32_t
        {
            Undefined = 0,
            U8 = 1,
            I8 = 2,
            U16 = 3,
            I16 = 4,
            U32_Color = 5,
            I32 = 6,
            U64_Color = 7,
            I64 = 8,
            Sgl = 9,
            Dbl = 10,
            ComplexSgl = 11,
            ComplexDbl = 12,
            Array1dU8 = 13,
            Array1dI8 = 14,
            Array1dU16 = 15,
            Array1dI16 = 16,
            Array1dU32_Color = 17,
            Array1dI32 = 18,
            Array1dU64_Color = 19,
            Array1dI64 = 20,
            Array1dSgl = 21,
            Array1dDbl = 22,
            Array1dComplexSgl = 23,
            Array1dComplexDbl = 24,
            Array2dU8 = 25,
            Array2dI8 = 26,
            Array2dU16 = 27,
            Array2dI16 = 28,
            Array2dU32 = 29,
            Array2dU32_Color = 30,
            Array2dI32 = 31,
            Array2dU64 = 32,
            Array2dU64_Color = 33,
            Array2dI64 = 34,
            Array2dSgl = 35,
            Array2dDbl = 36,
            Array2dComplexSgl = 37,
            Array2dComplexDbl = 38,
            ImageMonoU8 = 39,
            ImageMonoU16 = 40,
            ImageMonoI16 = 41,
            ImageMonoSgl = 42,
            ImageColorU32 = 43,
            ImageColorU64 = 44,
            ImageComplexSgl = 45,
            Array1d_2dPointI32 = 46,
            Array1d_2dPointSGL = 47,
			Array1d_2dPointDBL = 48,
            Array1d_3dPointI32 = 49,
            Array1d_3dPointSGL = 50,
            Array1d_3dPointDBL = 51,
            Array1d_4dPointI32 = 52,
            Array1d_4dPointSGL = 53,
            Array1d_4dPointDBL = 54
        };

        /**
         * @brief Struct to pass array/image info from LabVIEW.
         */
        struct MatInfo
        {
            DataType datatype;
            int32_t height;
            int32_t width;
            int32_t channels;
            int32_t lineWidth;
            uint8_t null; // Flag to indicate if the LV array/image is empty
        };

         /**
         * @brief Creates a cv::Mat header for LabVIEW image data without copying.
         */
        cv::Mat lvMatToCvMat(void* address, MatInfo* matInfo);

        /**
         * @brief Selects either the source or destination Mat based on LV info.
         */
        cv::Mat lvDstMatToCvMat(cv::Mat src, void* addressDst, MatInfo* matInfoDst);

    } // namespace mat

} // namespace lvi