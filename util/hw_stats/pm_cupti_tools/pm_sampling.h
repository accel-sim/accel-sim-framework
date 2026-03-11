//
// Copyright 2024 NVIDIA Corporation. All rights reserved
//

// System headers
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <unordered_map>

// CUPTI headers
#include "helper_cupti.h"
#include <cupti_target.h>
#include <cupti_pmsampling.h>
#include <cupti_profiler_target.h>
#include <cupti_profiler_host.h>

static int globalRangeIndex = 0;

struct SamplerRange
{
    size_t rangeIndex;
    uint64_t startTimestamp;
    uint64_t endTimestamp;
    std::unordered_map<std::string, double> metricValues;
};

class CuptiProfilerHost
{
    public:

    std::string m_chipName;
    std::vector<SamplerRange> m_samplerRanges;
    CUpti_Profiler_Host_Object* m_pHostObject = nullptr;

    CuptiProfilerHost() = default;
    ~CuptiProfilerHost() = default;

    void SetUp(std::string chipName, std::vector<uint8_t>& counterAvailibilityImage)
    {
        m_chipName = chipName;
        CUPTI_API_CALL(Initialize(counterAvailibilityImage));
    }

    void TearDown()
    {
        CUPTI_API_CALL(Deinitialize());
    }


    CUptiResult CreateConfigImage(std::vector<const char*> metricsList, std::vector<uint8_t>& configImage)
    {
        // Add metrics to config image
        {
            CUpti_Profiler_Host_ConfigAddMetrics_Params configAddMetricsParams {CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
            configAddMetricsParams.pHostObject = m_pHostObject;
            configAddMetricsParams.ppMetricNames = metricsList.data();
            configAddMetricsParams.numMetrics = metricsList.size();
            CUPTI_API_CALL(cuptiProfilerHostConfigAddMetrics(&configAddMetricsParams));
        }

        // Get Config image size and data
        {
            CUpti_Profiler_Host_GetConfigImageSize_Params getConfigImageSizeParams {CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
            getConfigImageSizeParams.pHostObject = m_pHostObject;
            CUPTI_API_CALL(cuptiProfilerHostGetConfigImageSize(&getConfigImageSizeParams));
            configImage.resize(getConfigImageSizeParams.configImageSize);

            CUpti_Profiler_Host_GetConfigImage_Params getConfigImageParams = {CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
            getConfigImageParams.pHostObject = m_pHostObject;
            getConfigImageParams.pConfigImage = configImage.data();
            getConfigImageParams.configImageSize = configImage.size();
            CUPTI_API_CALL(cuptiProfilerHostGetConfigImage(&getConfigImageParams));
        }

        // Get Num of Passes
        {
            CUpti_Profiler_Host_GetNumOfPasses_Params getNumOfPassesParam {CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
            getNumOfPassesParam.pConfigImage = configImage.data();
            getNumOfPassesParam.configImageSize = configImage.size();
            CUPTI_API_CALL(cuptiProfilerHostGetNumOfPasses(&getNumOfPassesParam));
            std::cout << "Num of Passes: " << getNumOfPassesParam.numOfPasses << "\n";
        }

        return CUPTI_SUCCESS;
    }

    CUptiResult EvaluateCounterData(CUpti_PmSampling_Object* pSamplingObject, size_t rangeIndex, std::vector<const char*> metricsList, std::vector<uint8_t>& counterDataImage)
    {
        m_samplerRanges.push_back(SamplerRange{});
        SamplerRange& samplerRange = m_samplerRanges.back();

        CUpti_PmSampling_CounterData_GetSampleInfo_Params getSampleInfoParams = {CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
        getSampleInfoParams.pPmSamplingObject = pSamplingObject;
        getSampleInfoParams.pCounterDataImage = counterDataImage.data();
        getSampleInfoParams.counterDataImageSize = counterDataImage.size();
        getSampleInfoParams.sampleIndex = rangeIndex;
        CUPTI_API_CALL(cuptiPmSamplingCounterDataGetSampleInfo(&getSampleInfoParams));

        samplerRange.startTimestamp = getSampleInfoParams.startTimestamp;
        samplerRange.endTimestamp = getSampleInfoParams.endTimestamp;

        std::vector<double> metricValues(metricsList.size());
        CUpti_Profiler_Host_EvaluateToGpuValues_Params evalauateToGpuValuesParams {CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
        evalauateToGpuValuesParams.pHostObject = m_pHostObject;
        evalauateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
        evalauateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
        evalauateToGpuValuesParams.ppMetricNames = metricsList.data();
        evalauateToGpuValuesParams.numMetrics = metricsList.size();
        evalauateToGpuValuesParams.rangeIndex = rangeIndex;
        evalauateToGpuValuesParams.pMetricValues = metricValues.data();
        CUPTI_API_CALL(cuptiProfilerHostEvaluateToGpuValues(&evalauateToGpuValuesParams));

        for (size_t i = 0; i < metricsList.size(); ++i) {
            samplerRange.metricValues[metricsList[i]] = metricValues[i];
        }

        return CUPTI_SUCCESS;
    }

    void PrintSampleRanges()
    {
        if (m_samplerRanges.empty())
        {
            std::cout << "No samples to print\n";
            return;
        }

        std::cout << "Total num of Samples: " << m_samplerRanges.size() << "\n";
        std::cout << "Printing first 50 samples:" << "\n";
        for(size_t sampleIndex = 0; sampleIndex < m_samplerRanges.size(); ++sampleIndex)
        {
            const auto& samplerRange = m_samplerRanges[sampleIndex];
            for (const auto& metric : samplerRange.metricValues)
            {
                std::cout << metric.second << ",";
            }
            std::cout << samplerRange.startTimestamp << "," << samplerRange.endTimestamp << "\n";
        }

    }

    void WriteCSVRanges(int deviceId = -1)
    {
        if (m_samplerRanges.empty())
        {
            std::cout << "No samples to print\n";
            return;
        }

        // Use globalRangeIndex to create a unique filename idx, with per-device prefix
        std::string fileName;
        if (deviceId >= 0) {
            fileName = "output_dev" + std::to_string(deviceId) + "_" + std::to_string(globalRangeIndex) + ".csv";
        } else {
            fileName = "output_" + std::to_string(globalRangeIndex) + ".csv";
        }
        std::ofstream outFile(fileName);
        globalRangeIndex++;
        if (!outFile.is_open())
        {
            std::cerr << "Failed to open " << fileName << " for writing\n";
            return;
        }

        // Write header
        if (!m_samplerRanges.empty())
        {
            const auto& firstRange = m_samplerRanges[0];
            for (const auto& metric : firstRange.metricValues)
            {
                outFile << metric.first << ",";
            }
            outFile << "StartTimestamp,EndTimestamp\n";
        }

        // Write data
        for (const auto& samplerRange : m_samplerRanges)
        {
            for (const auto& metric : samplerRange.metricValues)
            {
                outFile << metric.second << ",";
            }
            outFile << samplerRange.startTimestamp << "," << samplerRange.endTimestamp << "\n";
        }

        outFile.close();
        std::cout << "Data written to " << fileName << "\n\n";
    }

    void ClearRanges()
    {
        m_samplerRanges.clear();
    }

    CUptiResult GetSupportedBaseMetrics(std::vector<std::string>& metricsList)
    {
        for (size_t metricTypeIndex = 0; metricTypeIndex < CUPTI_METRIC_TYPE__COUNT; ++metricTypeIndex)
        {
            CUpti_Profiler_Host_GetBaseMetrics_Params getBaseMetricsParams {CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE};
            getBaseMetricsParams.pHostObject = m_pHostObject;
            getBaseMetricsParams.metricType = (CUpti_MetricType)metricTypeIndex;
            CUPTI_API_CALL(cuptiProfilerHostGetBaseMetrics(&getBaseMetricsParams));

            for (size_t metricIndex = 0; metricIndex < getBaseMetricsParams.numMetrics; ++metricIndex)
            {
                metricsList.push_back(getBaseMetricsParams.ppMetricNames[metricIndex]);
            }
        }
        return CUPTI_SUCCESS;
    }

    CUptiResult GetMetricProperties(const std::string& metricName, CUpti_MetricType& metricType, std::string& metricDescription)
    {
        CUpti_Profiler_Host_GetMetricProperties_Params getMetricPropertiesParams {CUpti_Profiler_Host_GetMetricProperties_Params_STRUCT_SIZE};
        getMetricPropertiesParams.pHostObject = m_pHostObject;
        getMetricPropertiesParams.pMetricName = metricName.c_str();
        CUPTI_API_CALL(cuptiProfilerHostGetMetricProperties(&getMetricPropertiesParams));
        metricType = getMetricPropertiesParams.metricType;
        metricDescription = getMetricPropertiesParams.pDescription;
        return CUPTI_SUCCESS;
    }

    CUptiResult GetSubMetrics(const std::string& metricName, std::vector<std::string>& subMetricsList)
    {
        CUpti_MetricType metricType;
        std::string metricDescription;
        CUPTI_API_CALL(GetMetricProperties(metricName, metricType, metricDescription));

        CUpti_Profiler_Host_GetSubMetrics_Params getSubMetricsParams {CUpti_Profiler_Host_GetSubMetrics_Params_STRUCT_SIZE};
        getSubMetricsParams.pHostObject = m_pHostObject;
        getSubMetricsParams.pMetricName = metricName.c_str();
        getSubMetricsParams.metricType = metricType;
        CUPTI_API_CALL(cuptiProfilerHostGetSubMetrics(&getSubMetricsParams));

        for (size_t subMetricIndex = 0; subMetricIndex < getSubMetricsParams.numOfSubmetrics; ++subMetricIndex)
        {
            subMetricsList.push_back(getSubMetricsParams.ppSubMetrics[subMetricIndex]);
        }
        return CUPTI_SUCCESS;
    }

private:
    CUptiResult Initialize(std::vector<uint8_t>& counterAvailibilityImage)
    {
        CUpti_Profiler_Host_Initialize_Params hostInitializeParams = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
        hostInitializeParams.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
        hostInitializeParams.pChipName = m_chipName.c_str();
        hostInitializeParams.pCounterAvailabilityImage = counterAvailibilityImage.data();
        CUPTI_API_CALL(cuptiProfilerHostInitialize(&hostInitializeParams));
        m_pHostObject = hostInitializeParams.pHostObject;
        return CUPTI_SUCCESS;
    }

    CUptiResult Deinitialize()
    {
        CUpti_Profiler_Host_Deinitialize_Params deinitializeParams = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        deinitializeParams.pHostObject = m_pHostObject;
        CUPTI_API_CALL(cuptiProfilerHostDeinitialize(&deinitializeParams));
        return CUPTI_SUCCESS;
    }
};


class CuptiPmSampling
{
    CUpti_PmSampling_Object* m_pmSamplerObject = nullptr;

public:
    CuptiPmSampling() = default;
    ~CuptiPmSampling() = default;

    void SetUp(int deviceIndex)
    {
        std::cout << "CUDA Device Number: " << deviceIndex << "\n";
        CUdevice cuDevice;
        DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceIndex));

        {
            int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
            DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
            DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
            std::cout << "Compute Capability of Device: " << computeCapabilityMajor << "." << computeCapabilityMinor << "\n";
            if (computeCapabilityMajor * 10 + computeCapabilityMinor < 75)
            {
                std::cerr << "Sample not supported as it requires compute capability >= 7.5\n";
                exit(0);
            }
        }

        CUPTI_API_CALL(InitializeProfiler());
    }

    void TearDown()
    {
        CUPTI_API_CALL(DeInitializeProfiler());
    }

    CUptiResult CreateCounterDataImage( uint64_t maxSamples, std::vector<const char*> metricsList, std::vector<uint8_t>& counterDataImage)
    {
        CUpti_PmSampling_GetCounterDataSize_Params getCounterDataSizeParams = {CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE};
        getCounterDataSizeParams.pPmSamplingObject = m_pmSamplerObject;
        getCounterDataSizeParams.numMetrics = metricsList.size();
        getCounterDataSizeParams.pMetricNames = metricsList.data();
        getCounterDataSizeParams.maxSamples  = maxSamples;
        CUPTI_API_CALL(cuptiPmSamplingGetCounterDataSize(&getCounterDataSizeParams));

        counterDataImage.resize(getCounterDataSizeParams.counterDataSize);
        CUpti_PmSampling_CounterDataImage_Initialize_Params initializeParams {CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
        initializeParams.pPmSamplingObject = m_pmSamplerObject;
        initializeParams.counterDataSize = counterDataImage.size();
        initializeParams.pCounterData = counterDataImage.data();
        CUPTI_API_CALL(cuptiPmSamplingCounterDataImageInitialize(&initializeParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult ResetCounterDataImage(std::vector<uint8_t>& counterDataImage)
    {
        CUpti_PmSampling_CounterDataImage_Initialize_Params initializeParams {CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
        initializeParams.pPmSamplingObject = m_pmSamplerObject;
        initializeParams.counterDataSize = counterDataImage.size();
        initializeParams.pCounterData = counterDataImage.data();
        CUPTI_API_CALL(cuptiPmSamplingCounterDataImageInitialize(&initializeParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult EnablePmSampling(size_t devIndex)
    {
        CUpti_PmSampling_Enable_Params enableParams {CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
        enableParams.deviceIndex = devIndex;
        CUPTI_API_CALL(cuptiPmSamplingEnable(&enableParams));
        m_pmSamplerObject = enableParams.pPmSamplingObject;
        return CUPTI_SUCCESS;
    }

    CUptiResult DisablePmSampling()
    {
        CUpti_PmSampling_Disable_Params disableParams {CUpti_PmSampling_Disable_Params_STRUCT_SIZE};
        disableParams.pPmSamplingObject = m_pmSamplerObject;
        CUPTI_API_CALL(cuptiPmSamplingDisable(&disableParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult SetConfig(std::vector<uint8_t>& configImage, size_t hardwareBufferSize, uint64_t samplingInterval)
    {
        CUpti_PmSampling_SetConfig_Params setConfigParams = {CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE};
        setConfigParams.pPmSamplingObject = m_pmSamplerObject;

        setConfigParams.configSize = configImage.size();
        setConfigParams.pConfig = configImage.data();

        setConfigParams.hardwareBufferSize = hardwareBufferSize;
        setConfigParams.samplingInterval = samplingInterval;

        setConfigParams.triggerMode = CUpti_PmSampling_TriggerMode::CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
        CUPTI_API_CALL(cuptiPmSamplingSetConfig(&setConfigParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult StartPmSampling()
    {
        CUpti_PmSampling_Start_Params startProfilingParams = {CUpti_PmSampling_Start_Params_STRUCT_SIZE};
        startProfilingParams.pPmSamplingObject = m_pmSamplerObject;
        CUPTI_API_CALL(cuptiPmSamplingStart(&startProfilingParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult StopPmSampling()
    {
        CUpti_PmSampling_Stop_Params stopProfilingParams = {CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
        stopProfilingParams.pPmSamplingObject = m_pmSamplerObject;
        CUPTI_API_CALL(cuptiPmSamplingStop(&stopProfilingParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult DecodePmSamplingData(std::vector<uint8_t>& counterDataImage)
    {
        CUpti_PmSampling_DecodeData_Params decodeDataParams = {CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE};
        decodeDataParams.pPmSamplingObject = m_pmSamplerObject;
        decodeDataParams.pCounterDataImage = counterDataImage.data();
        decodeDataParams.counterDataImageSize = counterDataImage.size();
        CUPTI_API_CALL(cuptiPmSamplingDecodeData(&decodeDataParams));
        return CUPTI_SUCCESS;
    }

    CUpti_PmSampling_Object* GetPmSamplerObject()
    {
        return m_pmSamplerObject;
    }

    static CUptiResult GetChipName(size_t deviceIndex, std::string& chipName)
    {
        CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

        CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
        getChipNameParams.deviceIndex = deviceIndex;
        CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
        chipName = getChipNameParams.pChipName;
        return CUPTI_SUCCESS;
    }

    static CUptiResult GetCounterAvailabilityImage(size_t deviceIndex, std::vector<uint8_t>& counterAvailibilityImage)
    {
        CUpti_PmSampling_GetCounterAvailability_Params getCounterAvailabilityParams { CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE };
        getCounterAvailabilityParams.deviceIndex = deviceIndex;
        CUPTI_API_CALL(cuptiPmSamplingGetCounterAvailability(&getCounterAvailabilityParams));

        counterAvailibilityImage.clear();
        counterAvailibilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
        getCounterAvailabilityParams.pCounterAvailabilityImage = counterAvailibilityImage.data();
        CUPTI_API_CALL(cuptiPmSamplingGetCounterAvailability(&getCounterAvailabilityParams));
        return CUPTI_SUCCESS;
    }

private:
    CUptiResult InitializeProfiler()
    {
        CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult DeInitializeProfiler()
    {
        CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = { CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
        return CUPTI_SUCCESS;
    }
};
