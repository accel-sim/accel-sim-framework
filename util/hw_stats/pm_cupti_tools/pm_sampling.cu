/*
 *  Copyright 2024 NVIDIA Corporation. All rights reserved
 *
 * This sample demonstrates the usage of the PM sampling feature in the CUDA Profiling Tools Interface (CUPTI).
 * There are two parts to this sample:
 * 1) Querying the available metrics for a chip and their properties.
 * 2) Collecting PM sampling data for a CUDA workload.
 *
 * The PM sampling feature allows users to collect sampling data at a specific interval for the CUDA workload
 * launched in the application.
 * For the continuous collection usecase, two separate threads are required:
 * 1. A main thread for launching the CUDA workload.
 * 2. A decode thread for decoding the collected data at a certain interval.
 *
 * The user is responsible for continuously calling the decode API, which frees up the hardware buffer for storing new data.
 *
 * In this sample, In the main thread the CUDA workload is launched. This workload is a simple vector addition implemented
 * in the `VectorAdd` kernel.
 *
 * The decode thread where we call the `DecodeCounterData` API. This API decodes the raw PM sampling data stored
 * in the hardware to a counter data image that the user has allocated.
 *
 */

#include <atomic>
#include <chrono>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <thread>

#ifdef _WIN32
#define strdup _strdup
#endif

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

#include "pm_sampling.h"

// Kernels
__global__
void vectorAdd(const int *pA, const int *pB, int *pC, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        pC[i] = pA[i] + pB[i];
    }
}

std::atomic<bool> stopDecodeThread(false);

const int NUM_OF_ELEMS = 4096*4096*2;
const int THREAD_PER_BLOCKS = 512;


class VectorLaunchWorkLoad
{
    int m_numOfElements;
    int m_threadsPerBlock, m_blocksPerGrid;
    size_t m_size;

    int *pDeviceA, *pDeviceB, *pDeviceC;
    std::vector<int> pHostA, pHostB, pHostC;

public:
    VectorLaunchWorkLoad(int numElements = NUM_OF_ELEMS, int threadsPerBlock = THREAD_PER_BLOCKS) :
        m_numOfElements(numElements), m_threadsPerBlock(threadsPerBlock)
    {
        m_size = m_numOfElements * sizeof(int);
        m_blocksPerGrid = (m_numOfElements + m_threadsPerBlock - 1) / m_threadsPerBlock;
        pHostA.resize(m_numOfElements);
        pHostB.resize(m_numOfElements);
        pHostC.resize(m_numOfElements);
    }

    ~VectorLaunchWorkLoad() {}

    void InitializeVector(std::vector<int>& pVector)
    {
        for (int i = 0; i < m_numOfElements; i++) {
            pVector[i] = i;
        }
    }

    void CleanUp()
    {
        // Free device memory.
        RUNTIME_API_CALL(cudaFree(pDeviceA));
        RUNTIME_API_CALL(cudaFree(pDeviceB));
        RUNTIME_API_CALL(cudaFree(pDeviceC));
    }

    void SetUp()
    {
        // Initialize input vectors
        InitializeVector(pHostA);
        InitializeVector(pHostB);
        std::fill(pHostC.begin(), pHostC.end(), 0);

        // Allocate vectors in device memory
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, m_size));
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, m_size));
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, m_size));

        // Copy vectors from host memory to device memory
        RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA.data(), m_size, cudaMemcpyHostToDevice));
        RUNTIME_API_CALL(cudaMemcpy(pDeviceB, pHostB.data(), m_size, cudaMemcpyHostToDevice));
    }

    void TearDown()
    {
        // Kernel Launch Verification
        // Copy result from device memory to host memory
        // pHostC contains the result in host memory
        RUNTIME_API_CALL(cudaMemcpy(pHostC.data(), pDeviceC, m_size, cudaMemcpyDeviceToHost));

        // Verify result
        int sum;
        for (int i = 0; i < m_numOfElements; ++i)
        {
            sum  = pHostA[i] + pHostB[i];
            if (pHostC[i] != sum)
            {
                fprintf(stderr, "Error: Result verification failed.\n");
                exit(EXIT_FAILURE);
            }
        }
        printf("Result verification passed.\n");
        CleanUp();
    }

    cudaError_t LaunchKernel()
    {
        vectorAdd<<<m_blocksPerGrid, m_threadsPerBlock>>>(pDeviceA, pDeviceB, pDeviceC, m_numOfElements);
        return cudaGetLastError();
    }
};

struct ParsedArgs
{
    bool isDeviceIndexSet = false;
    bool isChipNameSet = false;
    int deviceIndex = 0;
    int queryBaseMetrics = 0;
    int queryMetricProperties = 0;
    std::string chipName;
    uint64_t samplingInterval = 10000; // 100us
    size_t hardwareBufferSize = 512 * 1024 * 1024; // 512MB
    uint64_t maxSamples = 10000;
    std::vector<const char*> metrics =
    {
        "gr__cycles_active.avg",                            // Active Cycles
        "gr__cycles_elapsed.max",                           // Elapsed Cycles
        "gpu__time_duration.sum",                           // Duration
        "sm__inst_executed_realtime.avg.per_cycle_active",  // Inst Executed per Active Cycle
        "sm__cycles_active.avg"                             // SM Active Cycles
    };
};

ParsedArgs parseArgs(int argc, char *argv[]);
void PmSamplingDeviceSupportStatus(CUdevice device);
int PmSamplingCollection(std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args);
int PmSamplingQueryMetrics(std::string chipName, std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args);
void DecodeCounterData(
    std::vector<uint8_t>& counterDataImage,
    std::vector<const char*> metricsList,
    CuptiPmSampling& cuptiPmSamplingTarget,
    CuptiProfilerHost& pmSamplingHost,
    CUptiResult& result
);

int main(int argc, char *argv[])
{
    ParsedArgs args = parseArgs(argc, argv);
    DRIVER_API_CALL(cuInit(0));

    std::string chipName = args.chipName;
    std::vector<uint8_t> counterAvailibilityImage;
    if ((args.isDeviceIndexSet && args.deviceIndex >= 0) || !args.isChipNameSet)
    {
        CUdevice cuDevice;
        DRIVER_API_CALL(cuDeviceGet(&cuDevice, args.deviceIndex));
        PmSamplingDeviceSupportStatus(cuDevice);

        CuptiPmSampling::GetChipName(args.deviceIndex, chipName);
        CuptiPmSampling::GetCounterAvailabilityImage(args.deviceIndex, counterAvailibilityImage);
    }

    if (args.queryBaseMetrics || args.queryMetricProperties)
    {
        return PmSamplingQueryMetrics(chipName, counterAvailibilityImage, args);
    }
    else
    {
        return PmSamplingCollection(counterAvailibilityImage, args);
    }
    return 0;
}

int PmSamplingQueryMetrics(std::string chipName, std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args)
{
    CuptiProfilerHost pmSamplingHost;
    pmSamplingHost.SetUp(chipName, counterAvailibilityImage);

    if (args.queryBaseMetrics)
    {
        std::vector<std::string> baseMetrics;
        CUPTI_API_CALL(pmSamplingHost.GetSupportedBaseMetrics(baseMetrics));
        printf("Base Metrics:\n");
        for (const auto& metric : baseMetrics)
        {
            printf("  %s\n", metric.c_str());
        }
        return 0;
    }

    if (args.queryMetricProperties)
    {
        for (const auto& metricName : args.metrics)
        {
            std::vector<std::string> subMetrics;
            CUPTI_API_CALL(pmSamplingHost.GetSubMetrics(metricName, subMetrics));
            printf("Sub Metrics for %s:\n", metricName);
            for (const auto& metric : subMetrics) {
                printf("  %s\n", metric.c_str());
            }

            std::string metricDescription;
            CUpti_MetricType metricType;
            CUPTI_API_CALL(pmSamplingHost.GetMetricProperties(metricName, metricType,metricDescription));

            printf("Metric Description: %s\n", metricDescription.c_str());
            printf("Metric Type: %s\n", metricType == CUPTI_METRIC_TYPE_COUNTER ? "Counter" : (metricType == CUPTI_METRIC_TYPE_RATIO) ? "Ratio" : "Throughput");
            printf("\n");
        }
        return 0;
    }

    pmSamplingHost.TearDown();
    return 0;
}



int PmSamplingCollection(std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args)
{
    std::string chipName;
    CuptiPmSampling::GetChipName(args.deviceIndex, chipName);

    CuptiProfilerHost pmSamplingHost;
    pmSamplingHost.SetUp(chipName, counterAvailibilityImage);

    std::vector<uint8_t> configImage;
    CUPTI_API_CALL(pmSamplingHost.CreateConfigImage(args.metrics, configImage));

    CuptiPmSampling cuptiPmSamplingTarget;
    cuptiPmSamplingTarget.SetUp(args.deviceIndex);

    // 1. Enable PM sampling and set config for the PM sampling data collection.
    CUPTI_API_CALL(cuptiPmSamplingTarget.EnablePmSampling(args.deviceIndex));
    CUPTI_API_CALL(cuptiPmSamplingTarget.SetConfig(configImage, args.hardwareBufferSize, args.samplingInterval));

    // 2. Create counter data image
    std::vector<uint8_t> counterDataImage;
    CUPTI_API_CALL(cuptiPmSamplingTarget.CreateCounterDataImage(args.maxSamples, args.metrics, counterDataImage));

    VectorLaunchWorkLoad vectorWorkLoad;
    vectorWorkLoad.SetUp();

    CUptiResult threadFuncResult;
    // 3. Launch the decode thread
    std::thread decodeThread(DecodeCounterData, std::ref(counterDataImage), std::ref(args.metrics), std::ref(cuptiPmSamplingTarget), std::ref(pmSamplingHost), std::ref(threadFuncResult));

    auto joinDecodeThread = [&]() {
        stopDecodeThread = true;
        decodeThread.join();
        if (threadFuncResult != CUPTI_SUCCESS)
        {
            const char *errstr;
            cuptiGetResultString(threadFuncResult, &errstr);
            std::cerr << "DecodeCounterData Thread failed with error " << errstr << std::endl;
            return 1;
        }
        return 0;
    };

    // 4. Start the PM sampling and launch the CUDA workload
    CUPTI_API_CALL(cuptiPmSamplingTarget.StartPmSampling());
    stopDecodeThread = false;

    const size_t NUM_OF_ITERATIONS = 100;
    for (size_t ii = 0; ii < NUM_OF_ITERATIONS; ++ii)
    {
        cudaError_t result = vectorWorkLoad.LaunchKernel();
        if (result != cudaSuccess)
        {
            std::cerr << "Kernel launch failed " << cudaGetErrorString(result) << std::endl;
            return joinDecodeThread();
        }
    }
    cudaError_t errResult = cudaDeviceSynchronize();
    if (errResult != cudaSuccess)
    {
        std::cerr << "DeviceSync Failed " << cudaGetErrorString(errResult) << std::endl;
        return joinDecodeThread();
    }

    // 5. Stop the PM sampling and join the decode thread
    CUPTI_API_CALL(cuptiPmSamplingTarget.StopPmSampling());
    joinDecodeThread();

    // 6. Print the sample ranges for the collected metrics
    pmSamplingHost.PrintSampleRanges();

    // 7. Disable PM sampling for release all the resources allocated in CUPTI
    CUPTI_API_CALL(cuptiPmSamplingTarget.DisablePmSampling());

    // 8. Clean up
    cuptiPmSamplingTarget.TearDown();
    pmSamplingHost.TearDown();
    vectorWorkLoad.TearDown();
    return 0;
}

void DecodeCounterData( std::vector<uint8_t>& counterDataImage,
                        std::vector<const char*> metricsList,
                        CuptiPmSampling& cuptiPmSamplingTarget,
                        CuptiProfilerHost& pmSamplingHost,
                        CUptiResult& result)
{
    while (!stopDecodeThread)
    {
        const char *errstr;
        result = cuptiPmSamplingTarget.DecodePmSamplingData(counterDataImage);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "DecodePmSamplingData failed with error " << errstr << std::endl;
            return;
        }

        CUpti_PmSampling_GetCounterDataInfo_Params counterDataInfo {CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
        counterDataInfo.pCounterDataImage = counterDataImage.data();
        counterDataInfo.counterDataImageSize = counterDataImage.size();
        result = cuptiPmSamplingGetCounterDataInfo(&counterDataInfo);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "cuptiPmSamplingGetCounterDataInfo failed with error " << errstr << std::endl;
            return;
        }

        for (size_t sampleIndex = 0; sampleIndex < counterDataInfo.numCompletedSamples; ++sampleIndex)
        {
            pmSamplingHost.EvaluateCounterData(cuptiPmSamplingTarget.GetPmSamplerObject(), sampleIndex, metricsList, counterDataImage);
        }
        result = cuptiPmSamplingTarget.ResetCounterDataImage(counterDataImage);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "ResetCounterDataImage failed with error " << errstr << std::endl;
            return;
        }
    }
}

void PrintHelp()
{
    printf("Usage:\n");
    printf("  Query Metrics:\n");
    printf("    List Base Metrics : ./pm_sampling --device/-d <deviceIndex> --chip/-c <chipname> --queryBaseMetrics/-q\n");
    printf("    List submetrics   : ./pm_sampling --device/-d <deviceIndex> --chip/-c <chipname> --metrics/-m <metric1,metric2,...> --queryMetricsProp/-p\n");
    printf("  Note: when device index flag is passed, the chip name flag will be ignored.\n");
    printf("  PM Sampling:\n");
    printf("    Collection: ./pm_sampling --device/-d <deviceIndex> --samplingInterval/-i <samplingInterval> --maxsamples/-s <maxSamples in CounterDataImage> --hardwareBufferSize/-b <hardware buffer size> --metrics/-m <metric1,metric2,...>\n");
}

ParsedArgs parseArgs(int argc, char *argv[])
{
    ParsedArgs args;
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--device" || arg == "-d")
        {
            args.deviceIndex = std::stoi(argv[++i]);
            args.isDeviceIndexSet = true;
        }
        else if (arg == "--samplingInterval" || arg == "-i")
        {
            args.samplingInterval = std::stoull(argv[++i]);
        }
        else if (arg == "--maxsamples" || arg == "-s")
        {
            args.maxSamples = std::stoull(argv[++i]);
        }
        else if (arg == "--hardwareBufferSize" || arg == "-b")
        {
            args.hardwareBufferSize = std::stoull(argv[++i]);
        }
        else if (arg == "--chip" || arg == "-c")
        {
            args.chipName = std::string(argv[++i]);
            args.isChipNameSet = true;
        }
        else if (arg == "--queryBaseMetrics" || arg == "-q")
        {
            args.queryBaseMetrics = 1;
        }
        else if (arg == "--queryMetricsProp" || arg == "-p")
        {
            args.queryMetricProperties = 1;
        }
        else if (arg == "--metrics" || arg == "-m")
        {
            std::stringstream ss(argv[++i]);
            std::string metric;
            args.metrics.clear();
            while (std::getline(ss, metric, ','))
            {
                args.metrics.push_back(strdup(metric.c_str()));
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            PrintHelp();
            exit(EXIT_SUCCESS);
        }
        else
        {
            fprintf(stderr, "Invalid argument: %s\n", arg.c_str());
            PrintHelp();
            exit(EXIT_FAILURE);
        }
    }
    return args;
}

void PmSamplingDeviceSupportStatus(CUdevice device)
{
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = device;
    params.api = CUPTI_PROFILER_PM_SAMPLING;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Unable to profile on device " << device << ::std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tWSL is not supported" << ::std::endl;
        }

        exit(EXIT_WAIVED);
    }
}
