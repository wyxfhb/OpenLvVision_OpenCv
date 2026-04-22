#include "Blob.h"

using namespace cv;
using namespace lvi;
using namespace lvi::mat;
using namespace lvi::error;
using namespace lvi::edvr;
using namespace lvi::array;

#pragma region Create

_declspec(dllexport) void CV_DNN_BLOB_Create(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef)
{
	handleError(errorCluster, [&]() -> MgErr 
	{

		EdvrObject<cv::Mat> blobObj(edvrRef, new cv::Mat());

		return mgNoErr;
	});
}

#pragma endregion Create

#pragma region BlobFromImage

_declspec(dllexport) void CV_DNN_BLOB_BlobFromImage(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	void* addressImage, lvi::mat::MatInfo* matInfoImage,
	double scalefactor, lvi::point::PointI32* size,
	lvi::array::LV_1DArrayHandle<double_t> meanHandle,
	LVBoolean* swapRB, LVBoolean* crop,
	int32_t ddepth)
{
	handleError(errorCluster, [&]() -> MgErr 
	{
		Mat image = lvMatToCvMat(addressImage, matInfoImage);
		Array1D<double_t> mean(meanHandle);
		EdvrObject<cv::Mat> blobObj(edvrRef);

		if (image.channels() == 4) 
		{
			//moves data to opencv memory layout and converts
			cv::cvtColor(image, image, cv::COLOR_RGBA2RGB);
		}

		if (mean.size() >= 1) 
		{
			dnn::blobFromImage(image, *blobObj, scalefactor, Size(size->x, size->y), mean.toScalar(), static_cast<bool>(*swapRB), static_cast<bool>(*crop), ddepth);
		}
		else
		{
			dnn::blobFromImage(image, *blobObj, scalefactor, Size(size->x, size->y), Scalar(), static_cast<bool>(*swapRB), static_cast<bool>(*crop), ddepth);
		}

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_DNN_BLOB_BlobFromImages(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	lvi::array::LV_1DArrayHandle<uint64_t> adressesHandle,
	lvi::array::LV_1DArrayHandle<lvi::mat::MatInfo> matInfoImagesHandle,
	double scalefactor, lvi::point::PointI32* size,
	lvi::array::LV_1DArrayHandle<double_t> meanHandle,
	LVBoolean* swapRB, LVBoolean* crop,
	int32_t ddepth)
{
	handleError(errorCluster, [&]() -> MgErr {
		
		EdvrObject<cv::Mat> blobObj(edvrRef);
		Array1D<double_t> mean(meanHandle);
		Array1D<uint64_t> adresses(adressesHandle);
		Array1D<MatInfo> matInfoImages(matInfoImagesHandle);

		std::vector<cv::Mat> images;
		for (size_t i = 0; i < adresses.size(); i++)
		{
			cv::Mat image = lvMatToCvMat(reinterpret_cast<void*>(&adresses[i]), &matInfoImages[i]);

			if (image.channels() == 4)
			{
				//moves data to opencv memory layout and converts
				cv::cvtColor(image, image, cv::COLOR_RGBA2RGB);
			}

			images.push_back(image);
		}		

		if (mean.size() >= 1)
		{
			dnn::blobFromImages(images, *blobObj, scalefactor, Size(size->x, size->y), mean.toScalar(), static_cast<bool>(*swapRB), static_cast<bool>(*crop), ddepth);
		}
		else
		{
			dnn::blobFromImages(images, *blobObj, scalefactor, Size(size->x, size->y), Scalar(), static_cast<bool>(*swapRB), static_cast<bool>(*crop), ddepth);
		}

		return mgNoErr;
	});
}

#pragma endregion BlobFromImage

#pragma region Read

_declspec(dllexport) void CV_DNN_BLOB_ReadInfo(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	BlobInfo* blobInfo)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<cv::Mat> blobObj(edvrRef);		

		blobInfo->Dimensions = blobObj->size[0];
		blobInfo->Channels = blobObj->size[1];
		blobInfo->Height = blobObj->size[2];
		blobInfo->Width = blobObj->size[3];
		blobInfo->Type = blobObj->type();		

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_DNN_BLOB_ReadData_U8(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	lvi::array::LV_4DArrayHandle<uint8_t> nchwArrayHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<cv::Mat> blobObj(edvrRef);

		if(blobObj->type() != CV_8U)
		{
			throw std::runtime_error("Blob data type is not CV_8U");
		}

		Array4D<uint8_t> nchwArray(nchwArrayHandle);
		return nchwArray.copyFrom(*blobObj);
	});
}

_declspec(dllexport) void CV_DNN_BLOB_ReadData_SGL(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	lvi::array::LV_4DArrayHandle<float> nchwArrayHandle)
{
	handleError(errorCluster, [&]() -> MgErr 
	{
		EdvrObject<cv::Mat> blobObj(edvrRef);

		if (blobObj->type() != CV_32F) 
		{
			throw std::runtime_error("Blob data type is not CV_32F");
		}

		Array4D<float> nchwArray(nchwArrayHandle);
		return nchwArray.copyFrom(*blobObj);
	});
}

#pragma endregion Read