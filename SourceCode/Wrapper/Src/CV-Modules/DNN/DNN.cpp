#include "DNN.h"

using namespace cv;
using namespace lvi;
using namespace lvi::error;
using namespace lvi::array;

#pragma region Backends and Targets   

_declspec(dllexport) void CV_DNN_GetAvailableBackends(lvi::error::ErrorClusterPtr errorCluster,
	lvi::array::LV_1DArrayHandle<BackendTarget> backendArrayHandle)
{
	handleError(errorCluster, [&]() -> MgErr
	{

		Array1D<BackendTarget> backendsArr(backendArrayHandle);

		auto backends = dnn::getAvailableBackends();
		backendsArr.resize(backends.size());

		for (size_t i = 0; i < backends.size(); i++)
		{
			backendsArr[i].backend = backends[i].first;
			backendsArr[i].target = backends[i].second;
		}

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_DNN_GetAvailableTargets(lvi::error::ErrorClusterPtr errorCluster,
	lvi::array::LV_1DArrayHandle<uint16_t> targetArrayHandle,
	int32_t backend)
{
	handleError(errorCluster, [&]() -> MgErr
	{
			Array1D<uint16_t> targetArr(targetArrayHandle);

			auto targets = cv::dnn::getAvailableTargets(static_cast<cv::dnn::Backend>(backend));
			targetArr.resize(targets.size());

			for (size_t i = 0; i < targets.size(); i++)
			{
				targetArr[i] = static_cast<uint16_t>(targets[i]);
			}

			return mgNoErr;
	});
}

#pragma endregion Backends and Targets  

#pragma region NMS

_declspec(dllexport) void CV_DNN_NMSBoxes(lvi::error::ErrorClusterPtr errorCluster,
    lvi::array::LV_1DArrayHandle<LvRect2f> bboxesHandle,
    lvi::array::LV_1DArrayHandle<float> scoresHandle,
    float scoreThreshold, float nmsThreshold,
    lvi::array::LV_1DArrayHandle<int32_t> indicesHandle,
    float eta, int32_t topK)
{
    handleError(errorCluster, [&]() -> MgErr
    {
        Array1D<LvRect2f> bboxesArr(bboxesHandle);
        Array1D<float> scoresArr(scoresHandle);

        std::vector<cv::Rect2d> boxes(bboxesArr.size());
        for (int32_t i = 0; i < bboxesArr.size(); i++)
        {
            boxes[i] = cv::Rect2d(
                static_cast<double>(bboxesArr[i].x),
                static_cast<double>(bboxesArr[i].y),
                static_cast<double>(bboxesArr[i].width),
                static_cast<double>(bboxesArr[i].height));
        }

        std::vector<float> scores(scoresArr.data(), scoresArr.data() + scoresArr.size());

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, indices, eta, topK);

        Array1D<int32_t> lvIndices(indicesHandle);
        return lvIndices.copyFrom(indices);
    });
}

_declspec(dllexport) void CV_DNN_NMSBoxesRotated(lvi::error::ErrorClusterPtr errorCluster,
    lvi::array::LV_1DArrayHandle<LvRotatedRect> bboxesHandle,
    lvi::array::LV_1DArrayHandle<float> scoresHandle,
    float scoreThreshold, float nmsThreshold,
    lvi::array::LV_1DArrayHandle<int32_t> indicesHandle,
    float eta, int32_t topK)
{
    handleError(errorCluster, [&]() -> MgErr
    {
        Array1D<LvRotatedRect> bboxesArr(bboxesHandle);
        Array1D<float> scoresArr(scoresHandle);

        std::vector<cv::RotatedRect> boxes(bboxesArr.size());
        for (int32_t i = 0; i < bboxesArr.size(); i++)
        {
            boxes[i] = cv::RotatedRect(
                cv::Point2f(bboxesArr[i].cx, bboxesArr[i].cy),
                cv::Size2f(bboxesArr[i].width, bboxesArr[i].height),
                bboxesArr[i].angle);
        }

        std::vector<float> scores(scoresArr.data(), scoresArr.data() + scoresArr.size());

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, indices, eta, topK);

        Array1D<int32_t> lvIndices(indicesHandle);
        return lvIndices.copyFrom(indices);
    });
}

#pragma endregion NMS