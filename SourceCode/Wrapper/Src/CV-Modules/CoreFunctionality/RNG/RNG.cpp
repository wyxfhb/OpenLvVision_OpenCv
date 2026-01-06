#include "Rng.h"

using namespace cv;
using namespace lvi;
using namespace lvi::mat;
using namespace lvi::error;
using namespace lvi::edvr;


_declspec(dllexport) void CV_CORE_RNG_Create(ErrorClusterPtr errorCluster,
	ReferencePtr edvrRef, uint64 state)
{
	handleError(errorCluster, [&]() -> MgErr {

		EdvrObject<RNG> rng(edvrRef, new RNG(state));

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_RNG_Fill(lvi::error::ErrorClusterPtr errorCluster,
	void* addressDst, lvi::mat::MatInfo* matInfoDst,
	void* addressA, lvi::mat::MatInfo* matInfoA,
	void* addressB, lvi::mat::MatInfo* matInfoB,
	lvi::edvr::ReferencePtr edvrRef, int distType,
	LVBoolean* saturate)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<RNG> rng(edvrRef);
		Mat dst = lvMatToCvMat(addressDst, matInfoDst);
		Mat a = lvMatToCvMat(addressA, matInfoA);
		Mat b = lvMatToCvMat(addressB, matInfoB);

		rng->fill(dst, distType, a, b, static_cast<bool>(*saturate));

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_RNG_Gaussian(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef, double* number, double sigma)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<RNG> rng(edvrRef);

		*number = rng->gaussian(sigma);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_RNG_Next(ErrorClusterPtr errorCluster,
	ReferencePtr edvrRef, uint32_t* number)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<RNG> rng(edvrRef);

		*number = rng->next();

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_RNG_ReadState(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef, uint64_t* state)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<RNG> rng(edvrRef);

		*state = rng->state;

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_RNG_Uniform_Int(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef, int32_t* number,
	int32_t a, int32_t b)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<RNG> rng(edvrRef);

		*number = rng->uniform(a,b);

		return mgNoErr;
	});
}

_declspec(dllexport) void CV_CORE_RNG_Uniform_Float(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef, float32* number,
	float32 a, float32 b)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<RNG> rng(edvrRef);

		*number = rng->uniform(a, b);

		return mgNoErr;
		});
}

_declspec(dllexport) void CV_CORE_RNG_Uniform_Double(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef, float64* number,
	float64 a, float64 b)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<RNG> rng(edvrRef);

		*number = rng->uniform(a, b);

		return mgNoErr;
		});
}

_declspec(dllexport) void CV_CORE_RNG_WriteState(lvi::error::ErrorClusterPtr errorCluster,
	lvi::edvr::ReferencePtr edvrRef, uint64_t state)
{
	handleError(errorCluster, [&]() -> MgErr {
		EdvrObject<RNG> rng(edvrRef);

		rng->state = state;

		return mgNoErr;
	});
}