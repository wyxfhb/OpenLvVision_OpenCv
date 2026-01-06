#include "MatHandling.h"

using namespace lvi::mat;
using namespace cv;

Mat lvi::mat::lvMatToCvMat(void* address, MatInfo* matInfo)
{
    if (matInfo->null == 1)
    {
        //return empty matrix		
        return Mat();
    }

    int32_t height = matInfo->height;
    int32_t width = matInfo->width;
    int32_t lineWidth = matInfo->lineWidth;

    switch (matInfo->datatype)
    {
        // Use enum class scope
    case DataType::Array1dU8:
        return Mat(height, width, CV_8UC1, reinterpret_cast<uint8_t*>(address));
    case DataType::Array1dI8:
        return Mat(height, width, CV_8SC1, reinterpret_cast<int8_t*>(address));
    case DataType::Array1dU16:
        return Mat(height, width, CV_16UC1, reinterpret_cast<uint16_t*>(address));
    case DataType::Array1dI16:
        return Mat(height, width, CV_16SC1, reinterpret_cast<int16_t*>(address));
    case DataType::Array1dU32_Color:
        return Mat(height, width, CV_8UC4, reinterpret_cast<uint32_t*>(address));
    case DataType::Array1dI32:
        return Mat(height, width, CV_32SC1, reinterpret_cast<int32_t*>(address));
    case DataType::Array1dU64_Color:
        return Mat(height, width, CV_16UC4, reinterpret_cast<uint64_t*>(address));
    case DataType::Array1dSgl:
        return Mat(height, width, CV_32FC1, reinterpret_cast<float*>(address));
    case DataType::Array1dComplexSgl:
        return Mat(height, width, CV_32FC2, reinterpret_cast<float*>(address));
    case DataType::Array1dDbl:
        return Mat(height, width, CV_64FC1, reinterpret_cast<double*>(address));
    case DataType::Array1dComplexDbl:
        return Mat(height, width, CV_64FC2, reinterpret_cast<double*>(address));
    case DataType::Array2dU8:
        return Mat(height, width, CV_8UC1, reinterpret_cast<uint8_t*>(address));
    case DataType::Array2dI8:
        return Mat(height, width, CV_8SC1, reinterpret_cast<int8_t*>(address));
    case DataType::Array2dU16:
        return Mat(height, width, CV_16UC1, reinterpret_cast<uint16_t*>(address));
    case DataType::Array2dI16:
        return Mat(height, width, CV_16SC1, reinterpret_cast<int16_t*>(address));
    case DataType::Array2dU32_Color:
        return Mat(height, width, CV_8UC4, reinterpret_cast<uint32_t*>(address));
    case DataType::Array2dI32:
        return Mat(height, width, CV_32SC1, reinterpret_cast<int32_t*>(address));
    case DataType::Array2dU64_Color:
        return Mat(height, width, CV_16UC4, reinterpret_cast<uint64_t*>(address));
    case DataType::Array2dSgl:
        return Mat(height, width, CV_32FC1, reinterpret_cast<float*>(address));
    case DataType::Array2dComplexSgl:
        return Mat(height, width, CV_32FC2, reinterpret_cast<float*>(address));
    case DataType::Array2dDbl:
        return Mat(height, width, CV_64FC1, reinterpret_cast<double*>(address));
    case DataType::Array2dComplexDbl:
        return Mat(height, width, CV_64FC2, reinterpret_cast<double*>(address));
    case DataType::ImageMonoU8:
        return Mat(height, width, CV_8UC1, reinterpret_cast<uint8_t*>(*static_cast<uint64_t*>(address)), lineWidth);
    case DataType::ImageMonoU16:
        return Mat(height, width, CV_16UC1, reinterpret_cast<uint16_t*>(*static_cast<uint64_t*>(address)), lineWidth);
    case DataType::ImageMonoI16:
        return Mat(height, width, CV_16SC1, reinterpret_cast<int16_t*>(*static_cast<uint64_t*>(address)), lineWidth);
    case DataType::ImageMonoSgl:
        return Mat(height, width, CV_32FC1, reinterpret_cast<float*>(*static_cast<uint64_t*>(address)), lineWidth);
    case DataType::ImageColorU32:
        return Mat(height, width, CV_8UC4, reinterpret_cast<uint8_t*>(*static_cast<uint64_t*>(address)), lineWidth);
    case DataType::ImageColorU64:
        return Mat(height, width, CV_16UC4, reinterpret_cast<uint16_t*>(*static_cast<uint64_t*>(address)), lineWidth);
    case DataType::ImageComplexSgl:
        return Mat(height, width, CV_32FC2, reinterpret_cast<float*>(*static_cast<uint64_t*>(address)), lineWidth);
    case DataType::Array1d_2dPointI32:
        return Mat(height, width, CV_32SC2, reinterpret_cast<int32_t*>(address));
    case DataType::Array1d_2dPointSGL:
        return Mat(height, width, CV_32FC2, reinterpret_cast<float*>(address));
    case DataType::Array1d_2dPointDBL:
        return Mat(height, width, CV_64FC2, reinterpret_cast<double*>(address));
    case DataType::Array1d_3dPointI32:
        return Mat(height, width, CV_32SC3, reinterpret_cast<int32_t*>(address));
    case DataType::Array1d_3dPointSGL:
        return Mat(height, width, CV_32FC3, reinterpret_cast<float*>(address));
    case DataType::Array1d_3dPointDBL:
        return Mat(height, width, CV_64FC3, reinterpret_cast<double*>(address));
    case DataType::Array1d_4dPointI32:
        return Mat(height, width, CV_32SC4, reinterpret_cast<int32_t*>(address));
    case DataType::Array1d_4dPointSGL:
        return Mat(height, width, CV_32FC4, reinterpret_cast<float*>(address));
    case DataType::Array1d_4dPointDBL:
        return Mat(height, width, CV_64FC4, reinterpret_cast<double*>(address));
    default:
        throw std::runtime_error("Format not supported in lvMatToCvMat");
    }
}

Mat lvi::mat::lvDstMatToCvMat(Mat src, void* addressDst, MatInfo* matInfoDst)
{
    //NULL not possible in Labview
    if (matInfoDst->null == 1)
    {
        //point to source
        return src;
    }
    else
    {
        //point to destination image
        return lvMatToCvMat(addressDst, matInfoDst);
    }
}