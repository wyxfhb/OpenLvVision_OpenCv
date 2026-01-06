#include "PCA.h"

using namespace cv;
using namespace lvi;
using namespace lvi::mat;
using namespace lvi::error;
using namespace lvi::edvr;

#pragma region Create 

_declspec(dllexport) void CV_CORE_PCA_Create(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef)
{
	handleError(errorCluster, [&]() -> MgErr {

		EdvrObject<PCA> pcaObj(edvrRef, new PCA());

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_PCA_CreateMaxComponents(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	void* addressData, MatInfo* matInfoData,
	void* addressMean, MatInfo* matInfoMean,
	int32_t flags, int32_t maxComponents)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat data = lvMatToCvMat(addressData, matInfoData);
		Mat mean = lvMatToCvMat(addressMean, matInfoMean);

		if(!mean.empty())
		{
			EdvrObject<PCA> pcaObj(edvrRef, new PCA(data,mean,flags, maxComponents));
		}
		else
		{
			EdvrObject<PCA> pcaObj(edvrRef, new PCA(data, noArray(), flags, maxComponents));
		}	

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_PCA_CreateRetainedVariance(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	void* addressData, MatInfo* matInfoData,
	void* addressMean, MatInfo* matInfoMean,
	int32_t flags,  double retainedVariance)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat data = lvMatToCvMat(addressData, matInfoData);
		Mat mean = lvMatToCvMat(addressMean, matInfoMean);

		if (!mean.empty())
		{
			EdvrObject<PCA> pcaObj(edvrRef, new PCA(data, mean, flags, retainedVariance));
		}
		else
		{
			EdvrObject<PCA> pcaObj(edvrRef, new PCA(data, noArray(), flags, retainedVariance));
		}

		return mgNoErr;
	});
}

#pragma endregion Create 

#pragma region Compute 

_declspec(dllexport) void CV_CORE_PCA_ComputeMaxComponents(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	void* addressData, MatInfo* matInfoData,
	void* addressMean, MatInfo* matInfoMean,
	int32_t flags, int32_t maxComponents)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat data = lvMatToCvMat(addressData, matInfoData);
		Mat mean = lvMatToCvMat(addressMean, matInfoMean);
		EdvrObject<cv::PCA> pcaObj(edvrRef);

		if (!mean.empty())
		{
			(*pcaObj)(data, mean, flags, maxComponents);
		}
		else
		{
			(*pcaObj)(data, cv::noArray(), flags, maxComponents);
		}

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_PCA_ComputeRetainedVariance(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	void* addressData, MatInfo* matInfoData,
	void* addressMean, MatInfo* matInfoMean,
	int32_t flags, double retainedVariance)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat data = lvMatToCvMat(addressData, matInfoData);
		Mat mean = lvMatToCvMat(addressMean, matInfoMean);
		EdvrObject<cv::PCA> pcaObj(edvrRef);

		if (!mean.empty())
		{
			(*pcaObj)(data, mean, flags, retainedVariance);
		}
		else
		{
			(*pcaObj)(data, cv::noArray(), flags, retainedVariance);
		}

		return mgNoErr;
	});
}

#pragma endregion Compute 

#pragma region FileIO 

_declspec(dllexport) void CV_CORE_PCA_Read(
	lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	const char* filePath)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<cv::PCA> pcaObj(edvrRef);

		cv::FileStorage fs(filePath, cv::FileStorage::READ);
		pcaObj->read(fs.root());
		fs.release();

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_PCA_Write(
	lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	const char* filePath)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<cv::PCA> pcaObj(edvrRef);

		cv::FileStorage fs(filePath, cv::FileStorage::WRITE);
		pcaObj->write(fs);
		fs.release();

		return mgNoErr;
		});
}

#pragma endregion FileIO

#pragma region Read 

_declspec(dllexport) void CV_CORE_PCA_ReadEigenValues(
	lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	lvi::array::LV_1DArrayHandle<double_t> valuesHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<cv::PCA> pcaObj(edvrRef);
		lvi::array::Array1D<double_t> lvValues(valuesHandle);
		return lvValues.copyFrom(pcaObj->eigenvalues);
	});
}

_declspec(dllexport) void CV_CORE_PCA_ReadEigenVectors(
	lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	lvi::array::LV_2DArrayHandle<double_t> vectorsHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<cv::PCA> pcaObj(edvrRef);
		lvi::array::Array2D<double_t> lvVectors(vectorsHandle);
		return lvVectors.copyFrom(pcaObj->eigenvectors);
		});
}

_declspec(dllexport) void CV_CORE_PCA_ReadMean(
	lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	lvi::array::LV_2DArrayHandle<double_t> meanHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<cv::PCA> pcaObj(edvrRef);
		lvi::array::Array2D<double_t> lvMean(meanHandle);
		return lvMean.copyFrom(pcaObj->mean);
	});
}

_declspec(dllexport) void CV_CORE_PCA_ReadMeanSize(
	lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	int32_t* rows, int32_t* cols)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<cv::PCA> pcaObj(edvrRef);

		*rows = pcaObj->mean.rows;
		*cols = pcaObj->mean.cols;	

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_PCA_ReadEigenVectorsSize(
	lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	int32_t* rows, int32_t* cols)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<cv::PCA> pcaObj(edvrRef);

		*rows = pcaObj->eigenvectors.rows;
		*cols = pcaObj->eigenvalues.cols;

		return mgNoErr;
	});
}

#pragma endregion Read 

#pragma region Project 

_declspec(dllexport) void CV_CORE_PCA_BackProject(
	lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	void* addressData, MatInfo* matInfoData,
	void* addressResult, MatInfo* matInfoResult)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<cv::PCA> pcaObj(edvrRef);
		Mat data = lvMatToCvMat(addressData, matInfoData);
		Mat result = lvMatToCvMat(addressResult, matInfoResult);

		pcaObj->backProject(data, result);
		return mgNoErr;
		});
}

_declspec(dllexport) void CV_CORE_PCA_Project(
	lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef,
	void* addressData, MatInfo* matInfoData,
	void* addressResult, MatInfo* matInfoResult)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<cv::PCA> pcaObj(edvrRef);
		Mat data = lvMatToCvMat(addressData, matInfoData);
		Mat result = lvMatToCvMat(addressResult, matInfoResult);

		pcaObj->project(data, result);
		return mgNoErr;
	});
}

#pragma endregion Project 