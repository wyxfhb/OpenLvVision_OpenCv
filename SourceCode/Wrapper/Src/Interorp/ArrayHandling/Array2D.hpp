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
         * @brief A simple C++ wrapper class for a LabVIEW 2D Array Handle.
         * This class matches the style of Array1D.
         */
        template <typename T>
        class Array2D
        {
        private:
            LV_2DArrayHandle<T> m_handle;

        public:
            /**
             * @brief Constructs the wrapper from a LabVIEW handle.
             * @param handle The LV_2DArrayHandle<T> passed from LabVIEW.
             */
            explicit Array2D(LV_2DArrayHandle<T> handle) : m_handle(handle)
            {
                if (!m_handle)
                {
                    throw std::invalid_argument("Array2D: Handle pointer is null.");
                }

                // Ensure the handle content is initialized if passed as an empty/null handle
                if (*m_handle == nullptr)
                {
                    // 2 dimensions (int32) = 8 bytes
                    MgErr err = DSSetHandleSize(reinterpret_cast<UHandle>(m_handle), sizeof(int32_t) * 2);
                    if (err != mgNoErr || *m_handle == nullptr)
                    {
                        throw std::runtime_error("Array2D: Failed to initialize null array handle content.");
                    }
                    (*m_handle)->dimSizes[0] = 0;
                    (*m_handle)->dimSizes[1] = 0;
                }
            }

            /**
             * @brief Resizes the LabVIEW array to (rows x cols).
             * @return MgErr (mgNoErr on success, mFullErr on memory failure).
             */
            MgErr resize(int32_t rows, int32_t cols)
            {
                // 2 * 4-byte header for dimensions
                size_t headerSize = sizeof(int32_t) * 2;
                size_t dataSize = (size_t)rows * (size_t)cols * sizeof(T);
                size_t neededSize = headerSize + dataSize;

                MgErr err = DSSetHandleSize(reinterpret_cast<UHandle>(m_handle), neededSize);
                if (err != mgNoErr)
                {
                    return err;
                }

                if (*m_handle)
                {
                    (*m_handle)->dimSizes[0] = rows;
                    (*m_handle)->dimSizes[1] = cols;
                }

                return mgNoErr;
            }

            int32_t rows() const
            {
                return (*m_handle)->dimSizes[0];
            }

            int32_t cols() const
            {
                return (*m_handle)->dimSizes[1];
            }

            /**
             * @brief Returns a direct pointer to the array's data buffer.
             */
            T* data()
            {
                return (*m_handle)->data;
            }

            T& operator()(int32_t row, int32_t col)
            {
                int32_t width = (*m_handle)->dimSizes[1];
                return (*m_handle)->data[row * width + col];
            }

            /**
             * @brief Automatically resizes the LV array and copies data from a cv::Mat.
             * @param mat The source cv::Mat. Must match type T.
             */
            MgErr copyFrom(const cv::Mat& mat)
            {
                //Validate Type
                if (sizeof(T) != mat.elemSize())
                {
                    throw std::invalid_argument("Array2D: cv::Mat element size does not match array type T.");
                }

                int32_t r = mat.rows;
                int32_t c = mat.cols;

                // Resize LabVIEW Handle
                MgErr err = resize(r, c);
                if (err != mgNoErr) return err;

                if (r == 0 || c == 0) return mgNoErr;

                T* lvData = data();

                //Copy Data                
                if (mat.isContinuous())
                {
                    size_t totalBytes = (size_t)r * (size_t)c * sizeof(T);
                    std::memcpy(lvData, mat.ptr<T>(0), totalBytes);
                }
                else
                {
                    // Fallback: Copy row by row (handles ROIs and weird strides)
                    for (int i = 0; i < r; ++i)
                    {
                        const T* srcRow = mat.ptr<T>(i);
                        // LabVIEW rows are always packed (continuous)
                        T* dstRow = lvData + (i * c);
                       std::memcpy(dstRow, srcRow, c * sizeof(T));
                    }
                }

                return mgNoErr;
            }
        };

    } // namespace array
} // namespace lvi