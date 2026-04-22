#pragma once
#include "ArrayTypes.h"
#include "extcode.h"
#include <stdexcept>
#include <cstring>
#include <vector>
#include <type_traits>
#include <opencv2/core/mat.hpp>

namespace lvi
{
    namespace array
    {
        /**
         * @brief A simple C++ wrapper class for a LabVIEW 4D Array Handle.
         * Dimensions are NCHW (batch, channels, height, width).
         */
        template <typename T>
        class Array4D
        {
        private:
            LV_4DArrayHandle<T> m_handle;

        public:
            explicit Array4D(LV_4DArrayHandle<T> handle) : m_handle(handle)
            {
                if (!m_handle)
                {
                    throw std::invalid_argument("Array4D: Handle pointer is null.");
                }

                if (*m_handle == nullptr)
                {
                    // 4 dimensions (int32) = 16 bytes
                    MgErr err = DSSetHandleSize(reinterpret_cast<UHandle>(m_handle), sizeof(int32_t) * 4);
                    if (err != mgNoErr || *m_handle == nullptr)
                    {
                        throw std::runtime_error("Array4D: Failed to initialize null array handle content.");
                    }
                    (*m_handle)->dimSizes[0] = 0;
                    (*m_handle)->dimSizes[1] = 0;
                    (*m_handle)->dimSizes[2] = 0;
                    (*m_handle)->dimSizes[3] = 0;
                }
            }

            MgErr resize(int32_t d0, int32_t d1, int32_t d2, int32_t d3)
            {
                size_t headerSize = sizeof(int32_t) * 4;
                size_t dataSize = (size_t)d0 * (size_t)d1 * (size_t)d2 * (size_t)d3 * sizeof(T);
                size_t neededSize = headerSize + dataSize;

                MgErr err = DSSetHandleSize(reinterpret_cast<UHandle>(m_handle), neededSize);
                if (err != mgNoErr)
                {
                    return err;
                }

                if (*m_handle)
                {
                    (*m_handle)->dimSizes[0] = d0;
                    (*m_handle)->dimSizes[1] = d1;
                    (*m_handle)->dimSizes[2] = d2;
                    (*m_handle)->dimSizes[3] = d3;
                }

                return mgNoErr;
            }

            int32_t dim(int index) const
            {
                return (*m_handle)->dimSizes[index];
            }

            T* data()
            {
                return (*m_handle)->data;
            }

            T& operator()(int32_t n, int32_t c, int32_t h, int32_t w)
            {
                int32_t d1 = (*m_handle)->dimSizes[1];
                int32_t d2 = (*m_handle)->dimSizes[2];
                int32_t d3 = (*m_handle)->dimSizes[3];
                return (*m_handle)->data[n * (d1 * d2 * d3) + c * (d2 * d3) + h * d3 + w];
            }

            /**
             * @brief Copies data from a cv::Mat (NCHW) into the LV 4D array.
             * 2D [rows, cols]  -> NCHW [1, 1, rows, cols]
             * 3D [a, b, c]    -> NCHW [1, a, b, c]
             * 4D [a, b, c, d] -> NCHW [a, b, c, d]
             */
            MgErr copyFrom(const cv::Mat& mat)
            {
                if (!mat.isContinuous())
                {
                    throw std::invalid_argument("Array4D: cv::Mat is not continuous.");
                }

                // Right-align: pad higher (left) dimensions with 1
                int32_t d0, d1, d2, d3;
                switch (mat.dims)
                {
                    case 2:
                        d0 = 1; d1 = 1;
                        d2 = mat.size[0]; d3 = mat.size[1];
                        break;
                    case 3:
                        d0 = 1;
                        d1 = mat.size[0]; d2 = mat.size[1]; d3 = mat.size[2];
                        break;
                    case 4:
                        d0 = mat.size[0]; d1 = mat.size[1];
                        d2 = mat.size[2]; d3 = mat.size[3];
                        break;
                    default:
                        d0 = 1; d1 = 1; d2 = 1;
                        d3 = (mat.dims > 0) ? mat.size[0] : 1;
                        break;
                }

                MgErr err = resize(d0, d1, d2, d3);
                if (err != mgNoErr) return err;

                size_t totalBytes = mat.total() * sizeof(T);
                if (totalBytes > 0)
                {
                    std::memcpy(data(), mat.ptr<T>(0), totalBytes);
                }

                return mgNoErr;
            }
        };

    } // namespace array
} // namespace lvi
