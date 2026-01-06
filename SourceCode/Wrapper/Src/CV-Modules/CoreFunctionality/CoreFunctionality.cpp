#include "CoreFunctionality.h"

using namespace cv;
using namespace std;
using namespace lvi;
using namespace lvi::mat;
using namespace lvi::array;
using namespace lvi::point;
using namespace lvi::error;
using namespace lvi::edvr;

#pragma region ArithmeticOperations

_declspec(dllexport) void CV_CORE_AbsDiff(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	LV_1DArrayHandle<double_t> constantHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);
		Array1D<double_t> constantArray(constantHandle);

		if (!src2.empty())
		{ 
			cv::absdiff(src1, src2, dst); 
		}
		else {
			cv::absdiff(src1, constantArray.toScalar(), dst);
		}

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Add(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	LV_1DArrayHandle<double_t> constantHandle,
	void* addressMask, MatInfo* matInfoMask)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);
		Array1D<double_t> constantArray(constantHandle);

		if (!src2.empty())
		{
			cv::add(src1, src2, dst, mask); 
		}
		else 
		{
			cv::add(src1, constantArray.toScalar(), dst, mask);
		}

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_AddWeighted(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	double alpha, double beta, double gamma)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);

		cv::addWeighted(src1, alpha, src2, beta, gamma, dst);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_ConvertScaleAbs(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	double alpha, double beta)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);

		cv::convertScaleAbs(src1, dst, alpha, beta);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Divide(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	LV_1DArrayHandle<double_t> constantHandle, double scale)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);
		Array1D<double_t> constantArray(constantHandle);

		if (!src2.empty())
		{
			cv::divide(src1, src2, dst, scale); 
		}
		else 
		{
			cv::divide(src1, constantArray.toScalar(), dst, scale);
		}

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Exp(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);

		cv::exp(src, dst);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Log(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);

		cv::log(src, dst);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Multiply(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	LV_1DArrayHandle<double_t> constantHandle, double scale)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);
		Array1D<double_t> constantArray(constantHandle);

		if (!src2.empty())
		{
			cv::multiply(src1, src2, dst, scale); 
		}
		else 
		{
			cv::multiply(src1, constantArray.toScalar(), dst, scale);
		}
		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Pow(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	double power)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);

		cv::pow(src, power, dst);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_ScaleAdd(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	double alpha)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);

		cv::scaleAdd(src1, alpha, src2, dst);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Sqrt(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);

		cv::sqrt(src, dst);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Subtract(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	LV_1DArrayHandle<double_t> constantHandle,
	void* addressMask, MatInfo* matInfoMask)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);
		Array1D<double_t> constantArray(constantHandle);

		if (!src2.empty())
		{
			cv::subtract(src1, src2, dst, mask); 
		}
		else 
		{
			cv::subtract(src1, constantArray.toScalar(), dst, mask);
		}
		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Sum(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	LV_1DArrayHandle<double_t> valuesHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Array1D<double_t> valuesArray(valuesHandle);			

		return valuesArray.copyFrom(cv::sum(src));;
	});
}

#pragma endregion ArithmeticOperations

#pragma region BitwiseOperations
_declspec(dllexport) void CV_CORE_BitwiseAnd(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	void* addressMask, MatInfo* matInfoMask)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);

		cv::bitwise_and(src1, src2, dst, mask);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_BitwiseNot(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressMask, MatInfo* matInfoMask)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);

		cv::bitwise_not(src1, dst, mask);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_BitwiseOr(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	void* addressMask, MatInfo* matInfoMask)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);

		cv::bitwise_or(src1, src2, dst, mask);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_BitwiseXor(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	void* addressMask, MatInfo* matInfoMask)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);

		cv::bitwise_xor(src1, src2, dst, mask);

		return mgNoErr;
	});
}

#pragma endregion BitwiseOperations

#pragma region ChannelManipulation
_declspec(dllexport) void CV_CORE_ExtractChannel(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int32_t coi)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::extractChannel(src, dst, coi);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_InsertChannel(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int32_t coi)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::insertChannel(src, dst, coi);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_LUT(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	void* addressLut, MatInfo* matInfoLut)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);
		Mat lut = lvMatToCvMat(addressLut, matInfoLut);

		cv::LUT(src, lut, dst);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Merge(ErrorClusterPtr errorCluster,
	void* addressDst, MatInfo* matInfoDst,
	void* addressCH0, MatInfo* matInfoCH0,
	void* addressCH1, MatInfo* matInfoCH1,
	void* addressCH2, MatInfo* matInfoCH2)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);
		Mat ch0 = lvMatToCvMat(addressCH0, matInfoCH0);
		Mat ch1 = lvMatToCvMat(addressCH1, matInfoCH1);
		Mat ch2 = lvMatToCvMat(addressCH2, matInfoCH2);

		if (ch2.empty()) 
		{
			cv::merge(std::vector<Mat>{ ch0, ch1 }, dst); 
		}
		else 
		{
			Mat ch3(ch0.rows, ch0.cols, ch0.type(), Scalar(0));
			cv::merge(std::vector<Mat>{ ch0, ch1, ch2, ch3 }, dst); 
		}
		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_MixChannels(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int* fromTo, uint32_t size, LVBoolean* zeroChannels)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);
		if (*zeroChannels != LVFALSE && matInfoDst->null == 0) 
		{
			dst.setTo(cv::Scalar::all(0)); 
		}

		cv::mixChannels(src, dst, fromTo, size);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Split(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressCH0, MatInfo* matInfoCH0,
	void* addressCH1, MatInfo* matInfoCH1,
	void* addressCH2, MatInfo* matInfoCH2)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat ch0 = lvMatToCvMat(addressCH0, matInfoCH0);
		Mat ch1 = lvMatToCvMat(addressCH1, matInfoCH1);
		Mat ch2 = lvMatToCvMat(addressCH2, matInfoCH2);
		if (ch2.empty()) 
		{
			cv::split(src, std::vector<Mat>{ ch0, ch1 }); 
		}
		else 
		{
			Mat dummyAlpha(ch0.rows, ch0.cols, ch0.type());
			cv::split(src, std::vector<Mat>{ ch0, ch1, ch2, dummyAlpha}); 
		}
		return mgNoErr;
	});
}
#pragma endregion ChannelManipulation

#pragma region Comparison
_declspec(dllexport) void CV_CORE_CheckRange(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	LVBoolean* quiet, PointI32* pos,
	double minVal, double maxVal, LVBoolean* result)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		cv::Point cvPos;

		*result = cv::checkRange(src, static_cast<bool>(*quiet), &cvPos, minVal, maxVal);

		convertPoint(cvPos, pos);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Compare(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressSrc2, MatInfo* matInfoSrc2,
	void* addressDst, MatInfo* matInfoDst,
	LV_1DArrayHandle<double_t> constantHandle,
	cv::CmpTypes cmpType)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);
		Array1D<double_t> constantArray(constantHandle);

		if (!src2.empty())
		{
			cv::compare(src1, src2, dst, cmpType);
		}
		else
		{
			cv::compare(src1, constantArray.toScalar(), dst, cmpType);
		}

		return mgNoErr;

	});
}

_declspec(dllexport) void CV_CORE_InRange(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressLowerb, MatInfo* matLowerb,
	void* addressUpperb, MatInfo* matUpperb,
	void* addressDst, MatInfo* matInfoDst)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat lowerb = lvMatToCvMat(addressLowerb, matLowerb);
		Mat upperb = lvMatToCvMat(addressUpperb, matUpperb);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::inRange(src, lowerb, upperb, dst);

		return mgNoErr;
	});
}
#pragma endregion Comparison

#pragma region DataConcatenation
_declspec(dllexport) void CV_CORE_Hconcat(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::hconcat(src1, src2, dst);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Vconcat(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::vconcat(src1, src2, dst);

		return mgNoErr;
	});
}
#pragma endregion DataConcatenation

#pragma region FourierTransforms

_declspec(dllexport) void CV_CORE_Dct(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	LVBoolean* inverse,
	LVBoolean* rows)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);

		int flags = 0;

		if (*inverse != LVFALSE)
		{
			flags |= cv::DCT_INVERSE;
		}

		if (*rows != LVFALSE)
		{
			flags |= cv::DCT_ROWS;
		}

		cv::dct(src, dst,flags);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Dft_Complex(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	LVBoolean* rows,
	LVBoolean* scale)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		int flags = 0;
		if (rows && (*rows == LVBooleanTrue))
		{
			flags |= cv::DFT_ROWS;
		}
		if (scale && (*scale == LVBooleanTrue))
		{
			flags |= cv::DFT_SCALE;
		}

		cv::dft(src, dst, flags);

		return mgNoErr;
		});
}

_declspec(dllexport) void CV_CORE_GetOptimalDftSize(lvi::error::ErrorClusterPtr errorCluster,
	int32_t vecsize,
	int32_t* optimalSizeOut)
{
	handleError(errorCluster, [&]() -> MgErr {
		
		*optimalSizeOut = cv::getOptimalDFTSize(vecsize);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Dft_Real(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	LVBoolean* rows,
	LVBoolean* scale)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		int flags = cv::DFT_COMPLEX_OUTPUT;
		if (rows && (*rows == LVBooleanTrue))
		{
			flags |= cv::DFT_ROWS;
		}
		if (scale && (*scale == LVBooleanTrue))
		{
			flags |= cv::DFT_SCALE;
		}

		cv::dft(src, dst, flags);

		return mgNoErr;
		});
}

_declspec(dllexport) void CV_CORE_Idct(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	LVBoolean* rows)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src,addressDst, matInfoDst);

		int flags = 0;	

		if (*rows != LVFALSE)
		{
			flags |= cv::DCT_ROWS;
		}

		cv::idct(src, dst, flags);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_IDft_Complex(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	LVBoolean* rows,
	LVBoolean* scale)
{

	handleError(errorCluster, [&]() -> MgErr {

		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);


		int flags = 0;
		if (*rows != LVFALSE)
		{
			flags |= cv::DFT_ROWS;
		}

		if (*scale != LVFALSE)
		{
			flags |= cv::DFT_SCALE;
		}

		cv::idft(src, dst, flags);
		return mgNoErr;

	});
}

_declspec(dllexport) void CV_CORE_IDft_Real(lvi::error::ErrorClusterPtr errorCluster,

	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	LVBoolean* rows,
	LVBoolean* scale)
{
	handleError(errorCluster, [&]() -> MgErr {

		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		int flags = cv::DFT_REAL_OUTPUT;

		if (*rows != LVFALSE)
		{
			flags |= cv::DFT_ROWS;
		}

		if (*scale != LVFALSE)
		{
			flags |= cv::DFT_SCALE;
		}

		cv::idft(src, dst, flags);

		return mgNoErr;

	});
}

_declspec(dllexport) void CV_CORE_MulSpectrums(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrcA, MatInfo* matInfoSrcA,
	void* addressSrcB, MatInfo* matInfoSrcB,
	void* addressDst, MatInfo* matInfoDst,
	LVBoolean* rows, LVBoolean* conjB)
{
	handleError(errorCluster, [&]() -> MgErr {

		Mat srcA = lvMatToCvMat(addressSrcA, matInfoSrcA);
		Mat srcB = lvMatToCvMat(addressSrcB, matInfoSrcB);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		int flags = 0;
		if (*rows != LVFALSE)
		{
			flags |= cv::DFT_ROWS;
		}

		cv::mulSpectrums(srcA, srcB, dst, flags, static_cast<bool>(*conjB));

		return mgNoErr;
	});
}

#pragma endregion FourierTransforms

#pragma region GeometricOperations

_declspec(dllexport) void CV_CORE_CartToPolar(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressArray1, MatInfo* matInfoArray1,
	void* addressArray2, MatInfo* matInfoArray2,
	LVBoolean* angleInDegrees) 
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat array1 = lvMatToCvMat(addressArray1, matInfoArray1);
		Mat array2 = lvMatToCvMat(addressArray2, matInfoArray2);

		cv::cartToPolar(array1, array2, array1, array2, static_cast<bool>(*angleInDegrees));

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Magnitude(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressArray1, lvi::mat::MatInfo* matInfoArray1,
	void* addressArray2, lvi::mat::MatInfo* matInfoArray2)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat array1 = lvMatToCvMat(addressArray1, matInfoArray1);
		Mat array2 = lvMatToCvMat(addressArray2, matInfoArray2);

		cv::magnitude(array1, array2, array1);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Phase(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressArray1, MatInfo* matInfoArray1,
	void* addressArray2, MatInfo* matInfoArray2,
	LVBoolean* angleInDegrees)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat array1 = lvMatToCvMat(addressArray1, matInfoArray1);
		Mat array2 = lvMatToCvMat(addressArray2, matInfoArray2);

		cv::phase(array1, array2, array1, static_cast<bool>(*angleInDegrees));

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_PolarToCart(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressArray1, MatInfo* matInfoArray1,
	void* addressArray2, MatInfo* matInfoArray2,
	LVBoolean* angleInDegrees)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat array1 = lvMatToCvMat(addressArray1, matInfoArray1);
		Mat array2 = lvMatToCvMat(addressArray2, matInfoArray2);

		cv::polarToCart(array1, array2, array1, array2, static_cast<bool>(*angleInDegrees));

		return mgNoErr;
	});
}

#pragma endregion GeometricOperations

#pragma region LDA

_declspec(dllexport) void CV_CORE_LDA_SubspaceProject(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressW, MatInfo* matInfoW,
	void* addressMean, MatInfo* matInfoMean,
	void* addressResult, MatInfo* matInfoResult)
{
	handleError(errorCluster, [&]() -> MgErr {

		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat W = lvMatToCvMat(addressW, matInfoW);
		Mat mean = lvMatToCvMat(addressMean, matInfoMean);
		Mat dst = lvMatToCvMat(addressResult, matInfoResult);

		Mat result = cv::LDA::subspaceProject(W, mean, src);
		result.copyTo(dst);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_LDA_SubspaceReconstruct(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressProjected, MatInfo* matInfoProjected,
	void* addressW, MatInfo* matInfoW,
	void* addressMean, MatInfo* matInfoMean,
	void* addressResult, MatInfo* matInfoResult)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat projected = lvMatToCvMat(addressProjected, matInfoProjected);
		Mat W = lvMatToCvMat(addressW, matInfoW);
		Mat mean = lvMatToCvMat(addressMean, matInfoMean);
		Mat dst = lvMatToCvMat(addressResult, matInfoResult);

		Mat result = cv::LDA::subspaceReconstruct(W, mean, projected);
		result.copyTo(dst);

		return mgNoErr;
	});
}

#pragma endregion LDA

#pragma region LinearAlgebra
_declspec(dllexport) void CV_CORE_CompleteSymm(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
	LVBoolean* lowerToUpper)
{
	handleError(errorCluster, [&]() -> MgErr {
		
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		cv::completeSymm(src, static_cast<bool>(*lowerToUpper));

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Determinant(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	double* determinant)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);

		*determinant = cv::determinant(src);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Eigen(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
	LV_1DArrayHandle<double_t> eigenValuesHandle,
	void* addressEigenVectors, lvi::mat::MatInfo* matEigenVectors)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);		
		Mat eigenVectors = lvMatToCvMat(addressEigenVectors, matEigenVectors);
		Array1D<double_t> eigenValuesArray(eigenValuesHandle);
		Mat eigenValues;

		if(!eigenVectors.empty())
		{
			cv::eigen(src, eigenValues, eigenVectors);
		}
		else
		{
			cv::eigen(src, eigenValues);
		}

		
		return eigenValuesArray.copyFrom(eigenValues);
	});
}

_declspec(dllexport) void CV_CORE_EigenNonSymmetric(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
	LV_1DArrayHandle<double_t> eigenValuesHandle,
	void* addressEigenVectors, lvi::mat::MatInfo* matEigenVectors)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat eigenVectors = lvMatToCvMat(addressEigenVectors, matEigenVectors);
		Array1D<double_t> eigenValuesArray(eigenValuesHandle);
		Mat eigenValues;

		cv::eigenNonSymmetric(src, eigenValues, eigenVectors);

		return eigenValuesArray.copyFrom(eigenValues);
	});
}

_declspec(dllexport) void CV_CORE_Gemm(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
	void* addressDst, lvi::mat::MatInfo* matInfoDst,
	void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
	void* addressSrc3, lvi::mat::MatInfo* matInfoSrc3,
	double alpha,
	double beta,
	int flags)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat src3 = lvMatToCvMat(addressSrc3, matInfoSrc3);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);

		cv::gemm(src1, src2, alpha, src3, beta, dst, flags);

		return mgNoErr;
		});
}

_declspec(dllexport) void CV_CORE_Invert(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
	void* addressDst, lvi::mat::MatInfo* matInfoDst,
	double* value, int flags)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);

		*value = cv::invert(src, dst, flags);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_MulTransposed(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, lvi::mat::MatInfo* matInfoSrc,
	void* addressDst, lvi::mat::MatInfo* matInfoDst,
	void* addressDelta, lvi::mat::MatInfo* matInfoDelta,
	LVBoolean* aTa, double scale)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);		
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);
		Mat delta = lvMatToCvMat(addressDelta, matInfoDelta);

		if(!delta.empty())
		{
			cv::mulTransposed(src, dst, static_cast<bool>(*aTa), delta, scale);
		}
		else
		{
			cv::mulTransposed(src, dst, static_cast<bool>(*aTa), noArray(), scale);
		}	

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Norm(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
	void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
	void* addressMask, lvi::mat::MatInfo* matInfoMask,
	double* normValue, int normType)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);

		if(src2.empty())
		{
			*normValue = cv::norm(src1, normType, mask);
		}
		else
		{
			*normValue = cv::norm(src1, src2, normType, mask);
		}

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_PatchNaNs(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressA, lvi::mat::MatInfo* matInfoA,
	double value)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat a = lvMatToCvMat(addressA, matInfoA);

		cv::patchNaNs(a, value);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_SetIdentity(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressMtx, lvi::mat::MatInfo* matInfoMtx,
	LV_1DArrayHandle<double_t> sHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat mtx = lvMatToCvMat(addressMtx, matInfoMtx);
		Array1D<double_t> sArray(sHandle);

		cv::setIdentity(mtx, sArray.toScalar());

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Solve(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
	void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
	LV_2DArrayHandle<double_t> dstHandle,
	int32_t flags,
	LVBoolean* solutionFound)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat Src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat Src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);		
		Mat Dst;

		bool success = cv::solve(Src1, Src2, Dst, flags);

		if (success)
		{
			*solutionFound = LVBooleanTrue;
			Dst.convertTo(Dst, CV_64F);
			Array2D<double_t> lvDst(dstHandle);
			return lvDst.copyFrom(Dst);
		}
		else
		{
			*solutionFound = LVBooleanFalse;
		};

		return mgNoErr;
	});
}

__declspec(dllexport) void CV_CORE_SolveCubic(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressCoeffs, lvi::mat::MatInfo* matInfoCoeffs,
	lvi::array::LV_1DArrayHandle<double> rootsHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat coeffs = lvMatToCvMat(addressCoeffs, matInfoCoeffs);
		Mat roots;

		cv::solveCubic(coeffs, roots);

		roots.convertTo(roots, CV_64F);
		lvi::array::Array1D<double_t> lvRoots(rootsHandle);
		return lvRoots.copyFrom(roots);
	});
}

_declspec(dllexport) void CV_CORE_SolvePoly(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressCoeffs, lvi::mat::MatInfo* matInfoCoeffs,
	lvi::array::LV_1DArrayHandle<double_t> rootsHandle,
	int32_t maxIter)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat coeffs = lvMatToCvMat(addressCoeffs, matInfoCoeffs);
		Mat roots;
		
		cv::solvePoly(coeffs, roots, maxIter);
		roots.convertTo(roots, CV_64FC2);

		lvi::array::Array1D<double_t> lvRoots(rootsHandle);
		return lvRoots.copyFrom(roots);
	});
}

#pragma endregion LinearAlgebra

#pragma region MatrixImageTransformations

_declspec(dllexport) void CV_CORE_BorderInterpolate(
	lvi::error::ErrorClusterPtr errorCluster,	
	int32_t p, int32_t len, 
	int32_t borderType, int32_t* value)
{
	handleError(errorCluster, [&]() -> MgErr {
	
		*value = cv::borderInterpolate(p, len, borderType);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_CopyMakeBorder(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int32_t top, int32_t bottom, int32_t left, int32_t right,
	int32_t borderType,
	lvi::array::LV_1DArrayHandle<double_t> valueHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		cv::Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		cv::Mat dst = lvMatToCvMat(addressDst, matInfoDst);
		lvi::array::Array1D<double_t> values(valueHandle);

		cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, values.toScalar());

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Flip(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int32_t FlipCode)
{
	handleError(errorCluster, [&]() -> MgErr {
		cv::Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		cv::Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::flip(src, dst, FlipCode);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_FlipNd(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int32_t axis)
{
	handleError(errorCluster, [&]() -> MgErr {
		cv::Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		cv::Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::flipND(src, dst, axis);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_PerspectiveTransform(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressM, MatInfo* matInfoM)
{
	handleError(errorCluster, [&]() -> MgErr {
	
		cv::Mat src = lvMatToCvMat(addressSrc, matInfoSrc);	//1D Point array
		cv::Mat m = lvMatToCvMat(addressM, matInfoM);

		cv::perspectiveTransform(src, src , m);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Repeat(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int32_t ny, int32_t nx)
{
	handleError(errorCluster, [&]() -> MgErr {

		cv::Mat src = lvMatToCvMat(addressSrc, matInfoSrc);	
		cv::Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::repeat(src, ny, nx, dst);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Rotate(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int32_t rotateCode)
{
	handleError(errorCluster, [&]() -> MgErr {

		cv::Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		cv::Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::rotate(src, dst, rotateCode);
		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_TransformTo2D(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	void* addressM, MatInfo* matInfoM)
{
	handleError(errorCluster, [&]() -> MgErr {
		cv::Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		cv::Mat dst = lvMatToCvMat(addressDst, matInfoDst); //2D Points
		cv::Mat m = lvMatToCvMat(addressM, matInfoM);
		
		if (m.rows != 2)
		{
			throw std::runtime_error("Matrix: Must have 2 Rows");;
		}

		cv::transform(src, dst, m);
		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_TransformTo3D(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	void* addressM, MatInfo* matInfoM)
{
	handleError(errorCluster, [&]() -> MgErr {
		cv::Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		cv::Mat dst = lvMatToCvMat(addressDst, matInfoDst); //3D Points
		cv::Mat m = lvMatToCvMat(addressM, matInfoM);

		if (m.rows != 3)
		{
			throw std::runtime_error("Matrix: Must have 3 Rows");;
		}

		cv::transform(src, dst, m);
		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_TransformTo4D(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	void* addressM, MatInfo* matInfoM)
{
	handleError(errorCluster, [&]() -> MgErr {
		cv::Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		cv::Mat dst = lvMatToCvMat(addressDst, matInfoDst); //4D Points
		cv::Mat m = lvMatToCvMat(addressM, matInfoM);

		if (m.rows != 4)
		{
			throw std::runtime_error("Matrix: Must have 4 Rows");;
		}

		cv::transform(src, dst, m);
		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Transpose(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst)
{
	handleError(errorCluster, [&]() -> MgErr {

		cv::Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		cv::Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::transpose(src, dst);
		return mgNoErr;
	});
}

#pragma endregion MatrixImageTransformations

#pragma region PCA

_declspec(dllexport) void CV_CORE_PCACompute(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressData, MatInfo* matInfoData,
	lvi::array::LV_2DArrayHandle<double_t> meanHandle,
	lvi::array::LV_2DArrayHandle<double_t> eigenvectorsHandle,
	lvi::array::LV_1DArrayHandle<double_t> eigenvaluesHandle,
	int32_t maxComponents)
{
	handleError(errorCluster, [&]() -> MgErr {
		cv::Mat data = lvMatToCvMat(addressData, matInfoData);
		lvi::array::Array2D<double_t> lvMean(meanHandle);
		lvi::array::Array2D<double_t> lvVectors(eigenvectorsHandle);
		lvi::array::Array1D<double_t> lvValues(eigenvaluesHandle);

		cv::Mat mean;
		cv::Mat vectors;
		cv::Mat values;

		cv::PCACompute(data, mean, vectors, values, maxComponents);
		
		MgErr err = lvMean.copyFrom(mean);
		if (err != mgNoErr)
		{
			return err;
		}
		
		err = lvVectors.copyFrom(vectors);
		if(err != mgNoErr)
		{
			return err;
		}

		return lvValues.copyFrom(values);
	});
}

_declspec(dllexport) void CV_CORE_PCABackProject(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressData, MatInfo* matInfoData,
	void* addressMean, MatInfo* matInfoMean,
	void* addressEigVec, MatInfo* matInfoEigVec,
	void* addressResult, MatInfo* matInfoResult)
{
	handleError(errorCluster, [&]() -> MgErr {
		cv::Mat data = lvMatToCvMat(addressData, matInfoData);
		cv::Mat mean = lvMatToCvMat(addressMean, matInfoMean);
		cv::Mat eigVec = lvMatToCvMat(addressEigVec, matInfoEigVec);
		cv::Mat result = lvMatToCvMat(addressResult, matInfoResult);

		cv::PCABackProject(data, mean, eigVec, result);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_PCAProject(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressData, MatInfo* matInfoData,
	void* addressMean, MatInfo* matInfoMean,
	void* addressEigVec, MatInfo* matInfoEigVec,
	void* addressResult, MatInfo* matInfoResult)
{
	handleError(errorCluster, [&]() -> MgErr {
		cv::Mat data = lvMatToCvMat(addressData, matInfoData);
		cv::Mat mean = lvMatToCvMat(addressMean, matInfoMean);
		cv::Mat eigVec = lvMatToCvMat(addressEigVec, matInfoEigVec);
		cv::Mat result = lvMatToCvMat(addressResult, matInfoResult);
		
		cv::PCAProject(data, mean, eigVec, result);

		return mgNoErr;
	});
}

#pragma endregion PCA

#pragma region RandomNumber

_declspec(dllexport) void CV_CORE_Randn(ErrorClusterPtr errorCluster,
	void* addressDst, MatInfo* matInfoDst,
	void* addressMean, MatInfo* matInfoMean,
	void* addressStddev, MatInfo* matInfoStddev)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);
		Mat mean = lvMatToCvMat(addressMean, matInfoMean);
		Mat stddev = lvMatToCvMat(addressStddev, matInfoStddev);

		cv::randn(dst, mean, stddev);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_RandShuffle(lvi::error::ErrorClusterPtr errorCluster,
	void* addressDst, lvi::mat::MatInfo* matInfoDst,
	lvi::edvr::ReferencePtr edvrRef, double iterFactor)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);		

		//Check if no EDVR was provided
		if(*edvrRef != 0 )
		{
			EdvrObject<RNG> rng(edvrRef);
			cv::randShuffle(dst, iterFactor, rng.operator->());
		}
		else
		{
			cv::randShuffle(dst, iterFactor);
		}

		cv::randShuffle(dst, iterFactor);


		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Randu(ErrorClusterPtr errorCluster,
	void* addressDst, MatInfo* matInfoDst,
	void* addressLow, MatInfo* matInfoLow,
	void* addressHigh, MatInfo* matInfoHigh)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);
		Mat low = lvMatToCvMat(addressLow, matInfoLow);
		Mat high = lvMatToCvMat(addressHigh, matInfoHigh);

		cv::randu(dst, low, high);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_WriteRngSeed(lvi::error::ErrorClusterPtr errorCluster,
	int32 seed)
{
	handleError(errorCluster, [&]() -> MgErr {

		cv::setRNGSeed(seed);

		return mgNoErr;
	});
}


#pragma endregion RandomNumber

#pragma region Reduction

_declspec(dllexport) void CV_CORE_Reduce(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int32_t dim, cv::ReduceTypes rtype)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::reduce(src, dst, dim, rtype, CV_64F);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_ReduceArgMax(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int32_t axis, LVBoolean* lastIndex)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::reduceArgMax(src, dst, axis, static_cast<bool>(*lastIndex));

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_ReduceArgMin(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int32_t axis, LVBoolean* lastIndex)

	{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);

		cv::reduceArgMin(src, dst, axis, static_cast<bool>(*lastIndex));

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Trace(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	LV_1DArrayHandle<double_t> valuesHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Array1D<double_t> valuesArray(valuesHandle);		

		return valuesArray.copyFrom(cv::trace(src));
	});
}
#pragma endregion Reduction

#pragma region SimilarityMetrics

_declspec(dllexport) void CV_CORE_BatchDistance(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc1, lvi::mat::MatInfo* matInfoSrc1,
	void* addressSrc2, lvi::mat::MatInfo* matInfoSrc2,
	void* addressMask, lvi::mat::MatInfo* matInfoMask,
	lvi::array::LV_2DArrayHandle<float> distHandle,
	lvi::array::LV_2DArrayHandle<int32_t> nidxHandle,
	int32_t normType, int32_t K,
	int32_t update, LVBoolean* crosscheck)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);
		Mat dist;
		Mat nidx;

		cv::batchDistance(src1, src2, dist, CV_32F, nidx, normType, K, mask, update, static_cast<bool>(*crosscheck));

		lvi::array::Array2D<float> lvDist(distHandle);
		MgErr err = lvDist.copyFrom(dist);
		if (err != mgNoErr)
		{
			return err;
		}

		lvi::array::Array2D<int32_t> lvNidx(nidxHandle);
		return lvNidx.copyFrom(nidx);
		});
}


_declspec(dllexport) void CV_CORE_Mahalanobis(lvi::error::ErrorClusterPtr errorCluster,
	void* addressV1, MatInfo* matInfoV1,
	void* addressV2, MatInfo* matInfoV2,
	void* addressIcovar, MatInfo* matInfoIcovar,
	double_t* value) 
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat v1 = lvMatToCvMat(addressV1, matInfoV1);
		Mat v2 = lvMatToCvMat(addressV2, matInfoV2);
		Mat Icovar = lvMatToCvMat(addressIcovar, matInfoIcovar);

		*value = cv::Mahalanobis(v1, v2, Icovar);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_PSNR(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressSrc2, MatInfo* matInfoSrc2,
	double_t* value, double_t R)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);

		*value = cv::PSNR(src1, src2, R );

		return mgNoErr;
	});
}

#pragma endregion SimilarityMetrics

#pragma region Sort
_declspec(dllexport) void CV_CORE_Sort(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int sortFlags)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);

		cv::sort(src, dst, sortFlags);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_SortIdx(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	int sortFlags)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);

		cv::sortIdx(src, dst, sortFlags);

		return mgNoErr;
	});
}
#pragma endregion Sort

#pragma region Statistical
_declspec(dllexport) void CV_CORE_CountNonZero(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	int32_t* count)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);

		*count = cv::countNonZero(src);
		
		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_FindNonZero(lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	LV_1DArrayHandle<lvi::point::PointI32> locationsHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat locationsMat;
		Array1D<PointI32> lvArray(locationsHandle);

		cv::findNonZero(src, locationsMat);
		return lvArray.copyFrom(locationsMat);
	});
}

_declspec(dllexport) void CV_CORE_HasNonZero(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	LVBoolean* value)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);

		*value = cv::hasNonZero(src);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Max(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	double value)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);

		if (value == 0.0f)
		{
			cv::max(src1, src2, dst); 
		}
		else 
		{
			cv::max(src1, value, dst); 
		}

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Mean(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressMask, MatInfo* matInfoMask,
	LV_1DArrayHandle<double_t> meanHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);
		Array1D<double_t> meanArray(meanHandle);

		return meanArray.copyFrom(cv::mean(src, mask));
	});
}

_declspec(dllexport) void CV_CORE_MeanStdDev(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressMask, MatInfo* matInfoMask,
	LV_1DArrayHandle<double_t> meanHandle,
	LV_1DArrayHandle<double_t> stddevHandle)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);
		Array1D<double_t> meanArray(meanHandle);
		Array1D<double_t> stddevArray(stddevHandle);
			
		Scalar meanResult, stddevResult;
		cv::meanStdDev(src, meanResult, stddevResult, mask);

		MgErr err = meanArray.copyFrom(meanResult);
		if (err != mgNoErr)
		{
			return err;
		}
		err = stddevArray.copyFrom(stddevResult);
		return err;
	});
}


_declspec(dllexport) void CV_CORE_Min(ErrorClusterPtr errorCluster,
	void* addressSrc1, MatInfo* matInfoSrc1,
	void* addressDst, MatInfo* matInfoDst,
	void* addressSrc2, MatInfo* matInfoSrc2,
	double value)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src1 = lvMatToCvMat(addressSrc1, matInfoSrc1);
		Mat src2 = lvMatToCvMat(addressSrc2, matInfoSrc2);
		Mat dst = lvDstMatToCvMat(src1, addressDst, matInfoDst);

		if (value == 0.0f) 
		{
			cv::min(src1, src2, dst); 
		}
		else 
		{
			cv::min(src1, value, dst); 
		}
		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_MinMaxIdx(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressMask, MatInfo* matInfoMask,
	double* minVal, double* maxVal,
	int* minIdx, int* maxIdx)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);

		if (src.channels() > 1) 
		{
			cv::minMaxIdx(src, minVal, maxVal, nullptr, nullptr); 
		}
		else 
		{
			cv::minMaxIdx(src, minVal, maxVal, minIdx, maxIdx, mask); 
		}
		return mgNoErr;
		});
}

_declspec(dllexport) void CV_CORE_MinMaxLoc(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressMask, MatInfo* matInfoMask,
	double* minVal, double* maxVal,
	PointI32* minLoc, PointI32* maxLoc)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);

		if (src.channels() > 1) 
		{
			cv::minMaxLoc(src, minVal, maxVal, nullptr, nullptr); 
		}
		else 
		{
			cv::Point cvMinLoc, cvMaxLoc;
			cv::minMaxLoc(src, minVal, maxVal, &cvMinLoc, &cvMaxLoc, mask);

			convertPoint(cvMinLoc, minLoc);
			convertPoint(cvMaxLoc, maxLoc); 
		}
		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_Normalize(ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* matInfoSrc,
	void* addressDst, MatInfo* matInfoDst,
	void* addressMask, MatInfo* matInfoMask,
	double alpha, double beta, cv::NormTypes normType)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, matInfoSrc);
		Mat dst = lvDstMatToCvMat(src, addressDst, matInfoDst);
		Mat mask = lvMatToCvMat(addressMask, matInfoMask);

		cv::normalize(src, dst, alpha, beta, normType, -1, mask);

		return mgNoErr;
	});
}

#pragma endregion Statisticals

#pragma region SVD

_declspec(dllexport) void CV_CORE_SVBackSubst(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressW, MatInfo* infoW,
	void* addressU, MatInfo* infoU,
	void* addressVt, MatInfo* infoVt,
	void* addressRhs, MatInfo* infoRhs,
	void* addressDst, MatInfo* infoDst)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat w = lvMatToCvMat(addressW, infoW);
		Mat u = lvMatToCvMat(addressU, infoU);
		Mat vt = lvMatToCvMat(addressVt, infoVt);
		Mat rhs = lvMatToCvMat(addressRhs, infoRhs);
		Mat dst = lvMatToCvMat(addressDst, infoDst);
		 
		cv::SVD::backSubst(w, u, vt, rhs, dst);

		return mgNoErr;
		});
}

_declspec(dllexport) void CV_CORE_SVDecomp(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* infoSrc,
	void* addressW, MatInfo* infoW,
	void* addressU, MatInfo* infoU,
	void* addressVt, MatInfo* infoVt,
	int32_t flags)
{
	handleError(errorCluster, [&]() -> MgErr {

		Mat src = lvMatToCvMat(addressSrc, infoSrc);
		Mat w = lvMatToCvMat(addressW, infoW);
		Mat u = lvMatToCvMat(addressU, infoU);
		Mat vt = lvMatToCvMat(addressVt, infoVt);
		 
		cv::SVD::compute(src, w, u, vt, flags);
		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_SVD_Compute(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* infoSrc,
	void* addressW, MatInfo* infoW,
	int32_t flags)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, infoSrc);
		Mat w = lvMatToCvMat(addressW, infoW);

		cv::SVD::compute(src, w, flags);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_SVD_SolveZ(
	lvi::error::ErrorClusterPtr errorCluster,
	void* addressSrc, MatInfo* infoSrc,
	void* addressDst, MatInfo* infoDst)
{
	handleError(errorCluster, [&]() -> MgErr {
		Mat src = lvMatToCvMat(addressSrc, infoSrc);
		Mat dst = lvMatToCvMat(addressDst, infoDst);

		cv::SVD::solveZ(src, dst);

		return mgNoErr;
	});
}

#pragma endregion