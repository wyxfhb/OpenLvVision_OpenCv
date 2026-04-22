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
         * @brief A simple C++ wrapper class for a LabVIEW 1D Array Handle.
         * This class *borrows* the handle from LabVIEW, it does not own it.
         */
        template <typename T>
        class Array1D
        {
        private:
            LV_1DArrayHandle<T> m_handle;

            static constexpr size_t getHeaderSize()
            {
                size_t headerSize = sizeof(int32_t);
                size_t alignment = alignof(T);
                if (headerSize % alignment != 0) {
                    headerSize += alignment - (headerSize % alignment);
                }
                return headerSize;
            }

        public:
            /**
             * @brief Constructs the wrapper from a LabVIEW handle.
             * @param handle The LV_1DArrayHandle<T> passed from LabVIEW.
             */
            explicit Array1D(LV_1DArrayHandle<T> handle) : m_handle(handle)
            {
                if (!m_handle)
                {
                    throw std::invalid_argument("Array1D: Handle pointer (LStr**) is null.");
                }
                // Ensure the handle content is initialized
                if (*m_handle == nullptr)
                {
                    MgErr err = DSSetHandleSize(reinterpret_cast<UHandle>(m_handle), sizeof(int32_t));
                    if (err != mgNoErr || *m_handle == nullptr)
                    {
                        throw std::runtime_error("Array1D: Failed to initialize null array handle content.");
                    }
                    (*m_handle)->dimSize = 0; // Initialize size to 0
                }
            }            

            /**
             * @brief Resizes the LabVIEW array to hold newElementCount elements.
             * Handles both primitive types and LabVIEW pointers (like LStrHandle).
             * @param newElementCount The number of elements (T) the array should hold.
             * @return MgErr (mgNoErr on success).
             */
            MgErr resize(int32_t newElementCount)
            {
                int32_t oldElementCount = (*m_handle)->dimSize;

                // If sizes match, do nothing
                if (newElementCount == oldElementCount)
                {
                    return mgNoErr;
                }

                // If SHRINKING an array of handles (like LStrHandle), dispose of the dropped handles
                if constexpr (std::is_pointer_v<T>)
                {
                    if (newElementCount < oldElementCount)
                    {
                        for (int32_t i = newElementCount; i < oldElementCount; i++)
                        {
                            if ((*m_handle)->data[i] != nullptr)
                            {
                                DSDisposeHandle(reinterpret_cast<UHandle>((*m_handle)->data[i]));
                                (*m_handle)->data[i] = nullptr;
                            }
                        }
                    }
                }

                // Resize
                size_t neededSize = getHeaderSize() + (newElementCount * sizeof(T));
                MgErr err = DSSetHandleSize(reinterpret_cast<UHandle>(m_handle), neededSize);
                if (err != mgNoErr)
                {
                    return err;
                }

                // If GROWING, initialize the new memory
                if (newElementCount > oldElementCount)
                {
                    if constexpr (std::is_pointer_v<T>)
                    {
                        // For handles, allocate empty LabVIEW strings to prevent nullptr crashes
                        for (int32_t i = oldElementCount; i < newElementCount; i++)
                        {
                            UHandle newH = DSNewHandle(sizeof(int32_t));
                            if (newH)
                            {
                                *reinterpret_cast<int32_t*>(*newH) = 0;
                            }
                            (*m_handle)->data[i] = reinterpret_cast<T>(newH);
                        }
                    }
                    else
                    {
                        // For primitives (double, int), safe to zero out the memory.
                        std::memset((*m_handle)->data + oldElementCount, 0, (newElementCount - oldElementCount) * sizeof(T));
                    }
                }

                (*m_handle)->dimSize = newElementCount;
                return mgNoErr;
            }

            /**
             * @brief Returns the number of elements in the array.
             */
            int32_t size() const
            {
                return (*m_handle)->dimSize;
            }

            /**
             * @brief Returns a direct pointer to the array's data buffer.
             */
            T* data()
            {
                return (*m_handle)->data;
            }

            T& operator[](int32_t index)
            {
                return (*m_handle)->data[index];
            }

            const T& operator[](int32_t index) const
            {
                return (*m_handle)->data[index];
            }

            /**
             * @brief Copies the LV array contents into a std::vector.
             * Only valid for trivially copyable types.
             * @return A std::vector<T> containing a copy of the array data.
             */
            std::vector<T> toVector() const
            {
                static_assert(std::is_trivially_copyable_v<T>,
                    "toVector() only supports trivially copyable types.");

                int32_t n = (*m_handle)->dimSize;
                if (n <= 0) return {};
                return std::vector<T>((*m_handle)->data, (*m_handle)->data + n);
            }

            /**
             * @brief Automatically resizes the LV array and copies data from a std::vector.
             * @param vec The source vector to copy from.
             * @return MgErr (mgNoErr on success).
             */
            MgErr copyFrom(const std::vector<T>& vec)
            {
                // Ensure the data type is safe for direct memory copy
                static_assert(std::is_trivially_copyable_v<T>,
                    "copyFrom(vector) only supports trivially copyable types like int, double, or simple structs.");

                int32_t numElements = static_cast<int32_t>(vec.size());
                MgErr err = resize(numElements);
                if (err != mgNoErr)
                {
                    return err;
                }

                if (numElements > 0)
                {
                    std::memcpy(data(), vec.data(), numElements * sizeof(T));
                }
                return mgNoErr;
            }

            /**
             * @brief Copies data from a vector of vectors into a LV array-of-arrays.
             * Only valid when T is itself a LV_1DArrayHandle (i.e. Array1D<LV_1DArrayHandle<U>>).
             * Automatically resizes the outer array and copies each inner vector into its sub-array.
             * @tparam U The element type of the inner vectors.
             * @param vecs The source vector of vectors to copy from.
             * @return MgErr (mgNoErr on success).
             */
            template <typename U>
            MgErr copyFrom(const std::vector<std::vector<U>>& vecs)
            {
                static_assert(std::is_same_v<T, LV_1DArray<U>**>,
                    "copyFrom(vector<vector<U>>) is only valid for Array1D<LV_1DArrayHandle<U>>.");

                int32_t outerCount = static_cast<int32_t>(vecs.size());
                MgErr err = resize(outerCount);
                if (err != mgNoErr) return err;

                for (int32_t i = 0; i < outerCount; i++)
                {
                    Array1D<U> inner((*m_handle)->data[i]);
                    err = inner.copyFrom(vecs[i]);
                    if (err != mgNoErr) return err;
                }

                return mgNoErr;
            }

            /**
             * @brief Automatically resizes the LV array and copies data from a cv::Mat.
             * @param mat The source cv::Mat to copy from. Must be continuous and element size must match T.
             * @return MgErr (mgNoErr on success).
             */
            MgErr copyFrom(const cv::Mat& mat)
            {
                if (!mat.isContinuous())
                {
                    throw std::invalid_argument("cv::Mat is not continuous. Clone() it first.");
                }

                int32_t numDstElements = static_cast<int32_t>(mat.total());
                bool isSpecialCase = false;

                if constexpr (std::is_same_v<T, double>)
                {
					//Complex double special case
                    if (mat.type() == CV_64FC2)
                    {
                        numDstElements *= 2; // 1 Complex Element = 2 Double Elements
                        isSpecialCase = true;
                    }
                }

                if (!isSpecialCase)
                {
					//Default case: Ensure element size matches T
                    if (mat.elemSize() != sizeof(T))
                    {
                        throw std::invalid_argument("cv::Mat element size does not match array type T.");
                    }
                }

				//Resize LV array
                MgErr err = resize(numDstElements);
                if (err != mgNoErr)
                {
                    return err;
                }

				//Copy data
                if(numDstElements > 0)
                {
                    std::memcpy(data(), mat.data, numDstElements * sizeof(T));
                }

                return mgNoErr;
            }

             /**
             * @brief Automatically resizes the LV array to 4 elements
             * and copies data from a cv::Scalar.
             * @brief This function is only valid for Array1D<double>.
             * @param scalar The source cv::Scalar to copy from.
             * @return MgErr (mgNoErr on success).
             */
            MgErr copyFrom(const cv::Scalar& scalar)
            {
                static_assert(std::is_same_v<T, double>, "copyFrom(cv::Scalar) is only valid for Array1D<double>.");

                MgErr err = resize(4);
                if (err != mgNoErr)
                {
                    return err;
                }

                std::memcpy(data(), scalar.val, 4 * sizeof(double));

                return mgNoErr;
            }

            /**
             * @brief Converts the LV array (up to 4 elements) to a cv::Scalar.
             * @brief This function is only valid for Array1D<double>.
             * @return A cv::Scalar containing the first 1-4 elements from the array.
             */
            cv::Scalar toScalar() const
            {
                static_assert(std::is_same_v<T, double>,"toScalar() is only valid for Array1D<double>.");

                cv::Scalar scalar(0, 0, 0, 0);
                if (!m_handle || !(*m_handle))
                {
                    return scalar;
                }

                int32_t numElements = (*m_handle)->dimSize;
                T* pData = (*m_handle)->data;

                if (numElements >= 1) { scalar.val[0] = pData[0]; }
                if (numElements >= 2) { scalar.val[1] = pData[1]; }
                if (numElements >= 3) { scalar.val[2] = pData[2]; }
                if (numElements >= 4) { scalar.val[3] = pData[3]; }

                return scalar;
            }
        };

    } // namespace array
} // namespace lvi