#pragma once
#include "opencv2/core/types.hpp"

/**
* *@brief Namespace LabVIEW Interorp
*/
namespace lvi 
{ 
    /**
     * @brief Namespace for point conversion utilities
     */
    namespace point
    {

	    // --- Point Clusters for Labview ---
        struct PointI32 { int32_t x; int32_t y; };
        struct PointI64 { int64_t x; int64_t y; };
        struct PointSGL { float x; float y; };
        struct PointDBL { double x; double y; };

        /**
         * @brief A generic, templated function to convert any OpenCV-style
         * point (like cv::Point or cv::Point2f) to any LabVIEW-style
         * point (like PointI32 or PointSGL).
         *
         * @tparam T_CvPoint The OpenCV point type (e.g., cv::Point).
         * @tparam T_LvPoint The LabVIEW point type (e.g., PointI32).
         * @param p The input OpenCV point.
         * @param out A pointer to the output LabVIEW point.
         */
        template <typename T_CvPoint, typename T_LvPoint>
        void convertPoint(const T_CvPoint& p, T_LvPoint* out)
        {       
            out->x = static_cast<decltype(out->x)>(p.x);
            out->y = static_cast<decltype(out->y)>(p.y);
        }

    }// namespace points

} // namespace lvi
