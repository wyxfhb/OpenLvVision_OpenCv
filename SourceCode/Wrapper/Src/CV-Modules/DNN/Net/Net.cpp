#include "Net.h"

using namespace cv;
using namespace lvi;
using namespace lvi::mat;
using namespace lvi::error;
using namespace lvi::edvr;
using namespace lvi::array;
using namespace lvi::string;

#pragma region Create

_declspec(dllexport) void CV_DNN_NET_Create(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef)
{
    handleError(errorCluster, [&]() -> MgErr 
    {

        EdvrObject<cv::dnn::Net> netObj(edvrRef, new cv::dnn::Net());

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_ReadNet(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* modelPath, const char* configPath,
    const char* framework)
{
    handleError(errorCluster, [&]() -> MgErr 
    {
        cv::dnn::Net* net = new cv::dnn::Net(
            cv::dnn::readNet(std::string(modelPath),std::string(configPath),std::string(framework)));

        EdvrObject<cv::dnn::Net> netObj(edvrRef, net);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_ReadNetFromONNX(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* modelPath)
{
    handleError(errorCluster, [&]() -> MgErr 
    {
        cv::dnn::Net* net = new cv::dnn::Net(cv::dnn::readNetFromONNX(std::string(modelPath)));
        EdvrObject<cv::dnn::Net> netObj(edvrRef, net);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_ReadNetFromCaffe(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* prototxtPath, const char* caffeModelPath)
{
    handleError(errorCluster, [&]() -> MgErr 
    {
        cv::dnn::Net* net = new cv::dnn::Net(cv::dnn::readNetFromCaffe(std::string(prototxtPath),std::string(caffeModelPath)));
        EdvrObject<cv::dnn::Net> netObj(edvrRef, net);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_ReadNetFromTensorflow(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* modelPath, const char* configPath)
{
    handleError(errorCluster, [&]() -> MgErr 
    {
        cv::dnn::Net* net = new cv::dnn::Net(cv::dnn::readNetFromTensorflow(std::string(modelPath),std::string(configPath)));
        EdvrObject<cv::dnn::Net> netObj(edvrRef, net);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_ReadNetFromTFLite(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* modelPath)
{
    handleError(errorCluster, [&]() -> MgErr
    {
        cv::dnn::Net* net = new cv::dnn::Net(cv::dnn::readNetFromTFLite(std::string(modelPath)));
        EdvrObject<cv::dnn::Net> netObj(edvrRef, net);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_ReadNetFromTorch(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* modelPath, LVBoolean* isBinary,
    LVBoolean* evaluate)
{
    handleError(errorCluster, [&]() -> MgErr
    {
        cv::dnn::Net* net = new cv::dnn::Net(cv::dnn::readNetFromTorch(std::string(modelPath), static_cast<bool>(*isBinary), static_cast<bool>(*evaluate)));
        EdvrObject<cv::dnn::Net> netObj(edvrRef, net);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_ReadNetFromDarknet(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* cfgPath, const char* weightsPath)
{
    handleError(errorCluster, [&]() -> MgErr 
    {
        cv::dnn::Net* net = new cv::dnn::Net(cv::dnn::readNetFromDarknet(std::string(cfgPath),std::string(weightsPath)));
        EdvrObject<cv::dnn::Net> netObj(edvrRef, net);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_ReadNetFromModelOptimizer(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* xml, const char* bin)
{
    handleError(errorCluster, [&]() -> MgErr 
    {
        cv::dnn::Net* net = new cv::dnn::Net(cv::dnn::readNetFromModelOptimizer(std::string(xml),std::string(bin)));
        EdvrObject<cv::dnn::Net> netObj(edvrRef, net);

        return mgNoErr;
    });
}

#pragma endregion Create

#pragma region Forward

_declspec(dllexport) void CV_DNN_NET_Forward(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRefNet,    
    const char* outputName,
    lvi::array::LV_4DArrayHandle<float> resultHandle)
{
    handleError(errorCluster, [&]() -> MgErr 
    {
        EdvrObject<cv::dnn::Net> netObj(edvrRefNet);
        Array4D<float> lvResult(resultHandle);
               
        cv::Mat result = netObj->forward(std::string(outputName));

        return lvResult.copyFrom(result);
    });
}

_declspec(dllexport) void CV_DNN_NET_ForwardMultiple(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRefNet,
    lvi::array::LV_1DArrayHandle<lvi::array::LV_4DArrayHandle<float>> resultHandle,
    lvi::array::LV_1DArrayHandle<LStrHandle> namesHandles)
{
    handleError(errorCluster, [&]() -> MgErr 
    {
        EdvrObject<cv::dnn::Net> netObj(edvrRefNet);
        Array1D<LStrHandle> strings(namesHandles);
        std::vector<cv::String> names;

        for (size_t i = 0; i < strings.size(); i++)
        {
            std::string layerName;
            getLvStringHandle(strings[i], layerName);
			names.push_back(layerName);
        }

		std::vector <cv::Mat> blobs;
        netObj->forward(blobs,names);

        Array1D<lvi::array::LV_4DArrayHandle<float>> lvResult(resultHandle);
        MgErr err =  lvResult.resize(blobs.size());        

        if(err != mgNoErr)return err;

        for (size_t i = 0; i < blobs.size(); i++)
        {
            Array4D<float> blobResult(lvResult[i]);
            err = blobResult.copyFrom(blobs[i]);
        }

        return err;
    });
}


_declspec(dllexport) void CV_DNN_NET_SetInput(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRefNet,
    lvi::edvr::ReferencePtr edvrRefBlob,
    const char* name, double scalefactor,
	lvi::array::LV_1DArrayHandle<double> meanHandle)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRefNet);
        EdvrObject<cv::Mat> blobObj(edvrRefBlob);
        Array1D<double> mean(meanHandle);

        if(mean.size() > 0)
        {
            netObj->setInput(*blobObj, std::string(name),scalefactor,mean.toScalar());
        }
        else
        {
            netObj->setInput(*blobObj, std::string(name), scalefactor);;
		}             

        return mgNoErr;
    });
}

#pragma endregion Forward

#pragma region Configuration

_declspec(dllexport) void CV_DNN_NET_Connect(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* outPin, const char* inpPin)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

        netObj->connect(std::string(outPin), std::string(inpPin));

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_SetPreferableBackend(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    int32_t backendId)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

        netObj->setPreferableBackend(backendId);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_SetPreferableTarget(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    int32_t targetId)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

        netObj->setPreferableTarget(targetId);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_EnableFusion(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    LVBoolean* fusion)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

        netObj->enableFusion(static_cast<bool>(*fusion));

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_EnableWinograd(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    LVBoolean* useWinograd)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

        netObj->enableWinograd(static_cast<bool>(*useWinograd));

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_SetHalideScheduler(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* scheduler)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

        netObj->setHalideScheduler(std::string(scheduler));

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_SetInputShape(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* inputName,
    lvi::array::LV_1DArrayHandle<int32_t> shapeHandle)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);
        Array1D<int32_t> shapeArr(shapeHandle);

        cv::dnn::MatShape shape(shapeArr.data(), shapeArr.data() + shapeArr.size());
        netObj->setInputShape(std::string(inputName), shape);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_SetInputsNames(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    lvi::array::Array1D<LStrHandle> namesHandle)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);
        Array1D<LStrHandle> stringsArr(namesHandle);

        std::vector<cv::String> names;
        for (size_t i = 0; i < stringsArr.size(); i++)
        {
            std::string name;
            lvi::string::getLvStringHandle(stringsArr[i], name);
            names.push_back(name);
        }
        netObj->setInputsNames(names);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_RegisterOutput(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* outputName, int32_t layerId,
    int32_t outputPort, int32_t* boundLayer)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

       *boundLayer = netObj->registerOutput(std::string(outputName), layerId, outputPort);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_DumpToFile(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* path)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

        netObj->dumpToFile(std::string(path));

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_DumpToPbtxt(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* path)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

        netObj->dumpToPbtxt(std::string(path));

        return mgNoErr;
        });
}

#pragma endregion Configuration

#pragma region Layers

_declspec(dllexport) void CV_DNN_NET_Empty(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    LVBoolean* result)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

        *result = netObj->empty() ? LVBooleanTrue : LVFALSE;

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_GetLayerId(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* layer,
    int32_t* layerId)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

        *layerId = netObj->getLayerId(std::string(layer));

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_GetLayerNames(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    lvi::array::Array1D<LStrHandle> namesHandles)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef); 
        Array1D<LStrHandle> strings(namesHandles);

        std::vector<cv::String> names = netObj->getLayerNames();
        
		strings.resize(names.size());

        for (size_t i = 0; i < names.size(); i++)
        {
            setLvStringHandle(strings[i], names[i]);
        }

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_GetLayersCount(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    const char* type,
    int32_t* count)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

       *count = static_cast<int32_t>(netObj->getLayersCount(type));

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_GetLayerTypes(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    lvi::array::Array1D<LStrHandle> typesHandles)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);
        Array1D<LStrHandle> strings(typesHandles);

        std::vector<cv::String> types;
        netObj->getLayerTypes(types);
       
        strings.resize(types.size());

        for (size_t i = 0; i < types.size(); i++)
        {
            setLvStringHandle(strings[i], types[i]);
        }

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_GetLayerShapes(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    lvi::array::LV_1DArrayHandle<int32_t> netInputShapeHandle,
    int32_t layerId,
    lvi::array::LV_1DArrayHandle<LV_1DArrayHandle<int32_t>> inShapesHandle,
    lvi::array::LV_1DArrayHandle<LV_1DArrayHandle<int32_t>> outShapesHandle)
{
    handleError(errorCluster, [&]() -> MgErr 
    {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);
        Array1D<int32_t> inputShapeArr(netInputShapeHandle);

        cv::dnn::MatShape netInputShape = inputShapeArr.toVector();

        std::vector<cv::dnn::MatShape> inLayerShapes;
        std::vector<cv::dnn::MatShape> outLayerShapes;
        netObj->getLayerShapes(netInputShape, layerId, inLayerShapes, outLayerShapes);

        Array1D<LV_1DArrayHandle<int32_t>> lvInShapes(inShapesHandle);
        MgErr err = lvInShapes.copyFrom(inLayerShapes);
        if (err != mgNoErr) return err;

        Array1D<LV_1DArrayHandle<int32_t>> lvOutShapes(outShapesHandle);
        return lvOutShapes.copyFrom(outLayerShapes);
    });
}

_declspec(dllexport) void CV_DNN_NET_GetUnconnectedOutLayers(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    lvi::array::LV_1DArrayHandle<int32_t> layersHandle)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);
        Array1D<int32_t> layers(layersHandle);

        auto ids = netObj->getUnconnectedOutLayers();        	
        return layers.copyFrom(ids);
    });
}


_declspec(dllexport) void CV_DNN_NET_GetUnconnectedOutLayersNames(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    lvi::array::LV_1DArrayHandle<LStrHandle> namesHandles)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef); 
        Array1D<LStrHandle> strings(namesHandles);

        std::vector<cv::String> names = netObj->getUnconnectedOutLayersNames();
        
		strings.resize(names.size());

        for (size_t i = 0; i < names.size(); i++)
        {
           setLvStringHandle(strings[i], names[i]);           
        }

        return mgNoErr;
    });
}

#pragma endregion Layers

#pragma region Performance

_declspec(dllexport) void CV_DNN_NET_GetFLOPS(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    lvi::array::LV_1DArrayHandle<int32_t> netInputShapeHandle,
    int64_t* flops)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);
        Array1D<int32_t> inputShapeArr(netInputShapeHandle);

        cv::dnn::MatShape netInputShape = inputShapeArr.toVector();
        *flops = netObj->getFLOPS(netInputShape);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_GetMemoryConsumption(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    lvi::array::LV_1DArrayHandle<int32_t> netInputShapeHandle,
    int64_t* weights, int64_t* blobs)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);
        Array1D<int32_t> inputShapeArr(netInputShapeHandle);

        cv::dnn::MatShape netInputShape = inputShapeArr.toVector();

        size_t w = 0, b = 0;
        netObj->getMemoryConsumption(netInputShape, w, b);
        *weights = static_cast<int64_t>(w);
        *blobs = static_cast<int64_t>(b);

        return mgNoErr;
    });
}

_declspec(dllexport) void CV_DNN_NET_GetPerfProfile(lvi::error::ErrorClusterPtr errorCluster,
    lvi::edvr::ReferencePtr edvrRef,
    lvi::array::LV_1DArrayHandle<double> timingsHandle,
    int64_t* total)
{
    handleError(errorCluster, [&]() -> MgErr {
        EdvrObject<cv::dnn::Net> netObj(edvrRef);

        std::vector<double> timings;
        *total = netObj->getPerfProfile(timings);

        Array1D<double> lvTimings(timingsHandle);
        return lvTimings.copyFrom(timings);
    });
}

#pragma endregion Performance
