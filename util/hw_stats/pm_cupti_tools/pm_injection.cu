// Build as a shared library and set CUDA_INJECTION64_PATH to the .so
// Optional env vars:
//   INJECTION_METRICS="sm__cycles_elapsed.avg,smsp__sass_thread_inst_executed_op_dfma_pred_on.avg"
//   INJECTION_KERNEL_COUNT=10
//   PM_SAMPLING_INTERVAL_SYSCLK=200000    // GPU sysclk ticks between samples
//   PM_SAMPLING_HW_BUFFER_BYTES=1048576   // HW buffer per sampling session
//
// Copyright 2025

#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>
#include <cupti_target.h>
#include <cupti_activity.h>

#include <fstream>
#include <sstream>

#include "helper_cupti.h"

// Pull in your PM Sampling classes
#include "pm_sampling.h"  // provides CuptiPmSampling & CuptiProfilerHost (from your sample)

// Export symbol for CUDA injection
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__((visibility("default")))
#endif

using std::cerr;
using std::cout;
using std::endl;
using std::mutex;
using std::string;
using std::unordered_map;
using std::vector;

struct CtxData {
    CUcontext ctx{};
    int       deviceId{-1};

    // PM sampling components from your sample
    CuptiPmSampling     sampler;             // wraps cupti_pmsampling_* for target/device side
    CuptiProfilerHost   host;                // wraps host-side metric evaluation

    // Images/buffers
    vector<uint8_t> counterAvailabilityImage;
    vector<uint8_t> configImage;
    vector<uint8_t> counterDataImage;

    // metrics + limits
    vector<string> metricNamesOwned;         // keep storage alive
    vector<const char*> metricNamePtrs;      // CUpti wants const char*[]
    uint64_t samplingIntervalSysclk{200000}; // default if not set via env
    size_t   hwBufferBytes{1 << 20};         // 1 MiB default, can tune via env
    uint64_t maxSamples{16384};               // per session counter data capacity

    // session rotation
    int maxKernels{10};                      // INJECTION_KERNEL_COUNT
    int curKernels{0};
    int iterations{0};
    
    // kernel name tracking
    vector<string> kernelNames;              // store kernel names for current session

    bool samplingActive{false};

    // CUDA graph capture tracking: when true, kernel launch callbacks are recording
    // into a graph, not actually executing, so we must NOT start PM sampling.
    int captureDepth{0};
};

// global state
static mutex g_mutex;
static unordered_map<CUcontext, CtxData> g_ctx;

static vector<string> parseMetricsFromEnv()
{
    vector<string> metrics;
    if (const char* env = std::getenv("INJECTION_METRICS")) {
        // split on space, semicolon, comma
        const char* delims = " ,;";
        char* dup = strdup(env);
        for (char* tok = strtok(dup, delims); tok; tok = strtok(nullptr, delims)) {
            metrics.emplace_back(tok);
            cout << "Requesting metric '" << tok << "'\n";
        }
        free(dup);
    } else {
        // a couple of sane defaults (change as you prefer)
        metrics.push_back("sm__cycles_elapsed.avg");
        metrics.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.avg");
    }
    return metrics;
}

static void setupEnvTunables(CtxData& cd)
{
    if (const char* env = std::getenv("INJECTION_KERNEL_COUNT")) {
        int v = std::max(1, atoi(env));
        cd.maxKernels = v;
    }
    if (const char* env = std::getenv("PM_SAMPLING_INTERVAL_SYSCLK")) {
        uint64_t v = strtoull(env, nullptr, 10);
        if (v) cd.samplingIntervalSysclk = v;
    }
    if (const char* env = std::getenv("PM_SAMPLING_HW_BUFFER_BYTES")) {
        size_t v = strtoull(env, nullptr, 10);
        if (v) cd.hwBufferBytes = v;
    }
    if (const char* env = std::getenv("PM_SAMPLING_MAX_SAMPLES")) {
        uint64_t v = strtoull(env, nullptr, 10);
        if (v) cd.maxSamples = v;
    }
}

/* Still have to figure out how to print this in a non-invasive way
static void print_help()
{
    cout << "\n[PM-SAMPLING] Environment variables:\n"
            "  INJECTION_METRICS=\"gpc__cycles_elapsed.avg,sm__inst_executed.sum\"\n"
            "      Comma/space/semicolon-separated list of metrics to collect.\n"
            "      Default: gpc__cycles_elapsed.avg + sm__inst_executed.sum\n"
            "  INJECTION_KERNEL_COUNT=10\n"
            "      Number of kernels to collect per sampling session before rotating.\n"
            "      Default: 10\n"
            "  PM_SAMPLING_INTERVAL_SYSCLK=200000\n"
            "      GPU sysclk ticks between samples. Default: 200000\n"
            "  PM_SAMPLING_HW_BUFFER_BYTES=1048576\n"
            "      Size in bytes of HW buffer per sampling session. Default: 1048576 (1 MiB)\n"
            "  PM_SAMPLING_MAX_SAMPLES=8192\n"
            "      Maximum number of samples to store in counter data image per session. Default: 8192\n"
            "  PM_SAMPLING_CSV_PATH=\"path/to/file.csv\"\n"
            "      If set, write decoded sample data to this CSV file (appends).\n"
            "  PM_SAMPLING_CSV_TRUNCATE=1\n"
            "      If set to non-zero, truncate CSV file on first write.\n"
            "\n";
}
*/

// Check if PM sampling is supported on this device (requires compute capability >= 7.5)
static bool checkPmSamplingSupport(int deviceId)
{
    CUdevice cuDevice;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceId));

    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

    if (computeCapabilityMajor * 10 + computeCapabilityMinor < 75) {
        cerr << "[PM-SAMPLING] Error: PM Sampling requires compute capability >= 7.5, "
             << "but device " << deviceId << " has " << computeCapabilityMajor << "." << computeCapabilityMinor << endl;
        exit(1);
    }
    return true;
}

// initialize PM sampling for a context (once per context)
static void initContext(CUcontext ctx)
{
    CtxData cd{};
    cd.ctx = ctx;

    // determine current device
    RUNTIME_API_CALL(cudaGetDevice(&cd.deviceId));

    printf("[PM-SAMPLING] Initializing context %p on device %d for PM sampling\n",
           (void*)ctx, cd.deviceId);

    // Check compute capability before proceeding
    if (!checkPmSamplingSupport(cd.deviceId)) {
        return;  // Skip this context - PM sampling not supported
    }

    // get chip name + availability image using your helper wrappers
    {
        string chipName;
        CUPTI_API_CALL(CuptiPmSampling::GetChipName((size_t)cd.deviceId, chipName));
        CUPTI_API_CALL(CuptiPmSampling::GetCounterAvailabilityImage((size_t)cd.deviceId,
                                                                    cd.counterAvailabilityImage));

        // host-side profiler init bound to chip & availability image
        cd.host.SetUp(chipName, cd.counterAvailabilityImage); // creates host object
    }

    setupEnvTunables(cd);

    // metrics
    cd.metricNamesOwned = parseMetricsFromEnv();
    cd.metricNamePtrs.reserve(cd.metricNamesOwned.size());
    for (auto& s : cd.metricNamesOwned) cd.metricNamePtrs.push_back(s.c_str());

    // create config image for chosen metrics (host side), and size target counter data
    {
        CUPTI_API_CALL(cd.host.CreateConfigImage(cd.metricNamePtrs, cd.configImage)); // adds metrics & finalizes config
        // enable PM sampling (allocates sampler object)
        cd.sampler.SetUp(cd.deviceId);
        CUPTI_API_CALL(cd.sampler.EnablePmSampling((size_t)cd.deviceId));
        // allocate counter data image with capacity for samples
        CUPTI_API_CALL(cd.sampler.CreateCounterDataImage(cd.maxSamples, cd.metricNamePtrs, cd.counterDataImage));
        // set PM sampling config (interval/hw buffer; sysclk trigger per your sample)
        CUPTI_API_CALL(cd.sampler.SetConfig(cd.configImage, cd.hwBufferBytes, cd.samplingIntervalSysclk));
    }

    g_ctx[ctx] = std::move(cd);
}

// start sampling if not already active
static void startIfNeeded(CtxData& cd)
{
    if (!cd.samplingActive) {
        CUPTI_API_CALL(cd.sampler.StartPmSampling());
        cd.samplingActive = true;
        cd.iterations++;
    }
}

// Per-device CSV state so each MPI rank / GPU writes to its own file
struct PerDeviceCsvState {
    bool opened{false};
    bool wrote_header{false};
};
static unordered_map<int, PerDeviceCsvState> g_csvState;

static std::string getCsvPathForDevice(int deviceId) {
    const char* env = std::getenv("PM_SAMPLING_CSV_PATH");
    if (!env || !*env) return "";

    // Insert device ID before the extension: "output.csv" -> "output_dev0.csv"
    std::string base(env);
    auto dot = base.rfind('.');
    if (dot != std::string::npos) {
        return base.substr(0, dot) + "_dev" + std::to_string(deviceId) + base.substr(dot);
    }
    return base + "_dev" + std::to_string(deviceId);
}

static bool shouldTruncateCsvOnce() {
    static int once = -1;
    if (once == -1) {
        const char* v = std::getenv("PM_SAMPLING_CSV_TRUNCATE");
        once = (v && std::strcmp(v, "0") != 0) ? 1 : 0;
    }
    return once == 1;
}

static void writeCsvLines(int session, int deviceId, int kernelsInSession,
                          const std::string& captured)
{
    std::string path = getCsvPathForDevice(deviceId);
    if (path.empty()) return;

    auto& state = g_csvState[deviceId];

    // handle truncate-on-first-use
    std::ios_base::openmode mode = std::ios::out | std::ios::app;
    if (!state.opened) {
        if (shouldTruncateCsvOnce()) mode = std::ios::out; // overwrite once
        state.opened = true;
    }

    std::ofstream ofs(path, mode);
    if (!ofs) {
        std::cerr << "[PM-SAMPLING] Failed to open CSV path: " << path << "\n";
        return;
    }

    if (!state.wrote_header || mode == std::ios::out) {
        ofs << "session,device,kernels_in_session,line\n";
        state.wrote_header = true;
    }

    std::istringstream iss(captured);
    std::string line;
    while (std::getline(iss, line)) {
        // escape any embedded quotes by doubling them; wrap in quotes
        std::string quoted;
        quoted.reserve(line.size() + 2);
        quoted.push_back('"');
        for (char c : line) {
            if (c == '"') quoted.push_back('"');
            quoted.push_back(c);
        }
        quoted.push_back('"');

        ofs << session << "," << deviceId << "," << kernelsInSession << "," << quoted << "\n";
    }
}

// Per-device kernel name file state
struct PerDeviceKernelState {
    int globalKernelIndex{0};
    bool firstWrite{true};
};
static unordered_map<int, PerDeviceKernelState> g_kernelState;

static void writeKernelNamesFile(const vector<string>& kernelNames, int deviceId)
{
    auto& state = g_kernelState[deviceId];
    std::string fileName = "kernel_names_dev" + std::to_string(deviceId) + ".txt";
    std::ios_base::openmode mode = state.firstWrite ? std::ios::out : std::ios::app;
    std::ofstream ofs(fileName, mode);
    if (!ofs) {
        std::cerr << "[PM-SAMPLING] Failed to open kernel names file: " << fileName << "\n";
        return;
    }

    for (size_t i = 0; i < kernelNames.size(); ++i) {
        ofs << state.globalKernelIndex << "," << kernelNames[i] << "\n";
        state.globalKernelIndex++;
    }

    state.firstWrite = false;
    ofs.close();
}

// stop->decode->evaluate->print, then reset images for reuse
static void flushSession(CtxData& cd, const char* reason)
{
    if (!cd.samplingActive) return;

    cout << "[PM-SAMPLING] Flushing session " << cd.iterations
         << " on device " << cd.deviceId << " (" << reason << ")\n";

    CUPTI_API_CALL(cd.sampler.StopPmSampling());
    cd.samplingActive = false;

    // decode raw samples into counterDataImage
    CUPTI_API_CALL(cd.sampler.DecodePmSamplingData(cd.counterDataImage));

    // evaluate & print (first 50 samples for brevity, like your sample)
    {
        // Evaluate in a loop just like before, but we’ll capture the host’s pretty-print.
        size_t maxTry = cd.maxSamples; // cap; adjust if you want
        size_t ok = 0;
        for (size_t i = 0; i < maxTry; ++i) {
            CUptiResult r = cd.host.EvaluateCounterData(cd.sampler.GetPmSamplerObject(),
                                                        i, cd.metricNamePtrs, cd.counterDataImage);
            if (r != CUPTI_SUCCESS) break;
            ok++;
        }
        std::cout << "Decoded " << ok << " samples.\n";

        // Capture the host’s PrintSampleRanges() output
        std::ostringstream capture;
        auto* oldbuf = std::cout.rdbuf(capture.rdbuf());
        // cd.host.PrintSampleRanges();     // <-- prints into 'capture'
        cd.host.WriteCSVRanges(cd.deviceId);      // <-- Nice CSV write
        std::cout.rdbuf(oldbuf);         // restore

        // Tee to console
        std::cout << capture.str();

        // Optional CSV
        writeCsvLines((int)cd.iterations, cd.deviceId, cd.curKernels, capture.str());
        
        // Write kernel names file
        if (!cd.kernelNames.empty()) {
            writeKernelNamesFile(cd.kernelNames, cd.deviceId);
        }
    }
    // reuse the same buffers for next session
    CUPTI_API_CALL(cd.sampler.ResetCounterDataImage(cd.counterDataImage));
    cd.host.ClearRanges();
    cd.curKernels = 0;
    cd.kernelNames.clear();
}

// application exit hook
static void atexitHandler()
{
    CUPTI_API_CALL(cuptiGetLastError());

    // Try to acquire lock, but don't block if already locked
    if (g_mutex.try_lock()) {
        for (auto& kv : g_ctx) {
            auto& cd = kv.second;
            if (cd.samplingActive || cd.curKernels > 0) {
                flushSession(cd, "process-exit");
            }
        }
        // Clear the map without calling DisablePmSampling/TearDown.
        // The CUDA runtime's own atexit handlers may have already torn down
        // the underlying CUPTI/nvperf objects by this point, so calling into
        // them causes a double-free inside libnvperf_host.so.
        g_ctx.clear();
        g_mutex.unlock();
    } else {
        std::cerr << "[PM-SAMPLING] Warning: Could not acquire mutex during exit cleanup\n";
    }
    // If we can't get the lock, skip cleanup to avoid deadlock
}

// Helper: check if a driver API callback ID is a kernel launch variant
static bool isKernelLaunchCbid(CUpti_CallbackId cbid)
{
    return cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel
        || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel
        || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx;
}

// CUPTI callback
static void CUPTIAPI callback(
    void* /*userData*/,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const void* pCallbackData)
{
    if (domain == CUPTI_CB_DOMAIN_RESOURCE &&
        cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)
    {
        const CUpti_ResourceData* r = static_cast<const CUpti_ResourceData*>(pCallbackData);
        std::lock_guard<mutex> lk(g_mutex);
        if (!g_ctx.count(r->context)) {
            initContext(r->context);  // set up PM sampling infra for this context
        }
        return;
    }

    if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;

    // --- Track CUDA graph stream capture state ---
    // During capture, kernel launches are recorded into a graph and do NOT
    // actually execute, so we must skip PM sampling.
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2)
    {
        const CUpti_CallbackData* d = static_cast<const CUpti_CallbackData*>(pCallbackData);
        if (d->callbackSite == CUPTI_API_ENTER) {
            std::lock_guard<mutex> lk(g_mutex);
            auto it = g_ctx.find(d->context);
            if (it != g_ctx.end()) {
                it->second.captureDepth++;
                cout << "[PM-SAMPLING] Stream capture BEGIN (depth="
                     << it->second.captureDepth << ") on device "
                     << it->second.deviceId << "\n";
            }
        }
        return;
    }

    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture)
    {
        const CUpti_CallbackData* d = static_cast<const CUpti_CallbackData*>(pCallbackData);
        if (d->callbackSite == CUPTI_API_EXIT) {
            std::lock_guard<mutex> lk(g_mutex);
            auto it = g_ctx.find(d->context);
            if (it != g_ctx.end() && it->second.captureDepth > 0) {
                it->second.captureDepth--;
                cout << "[PM-SAMPLING] Stream capture END (depth="
                     << it->second.captureDepth << ") on device "
                     << it->second.deviceId << "\n";
            }
        }
        return;
    }

    // --- cuGraphLaunch: the graph's kernels execute here ---
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch)
    {
        const CUpti_CallbackData* d = static_cast<const CUpti_CallbackData*>(pCallbackData);
        std::lock_guard<mutex> lk(g_mutex);
        auto it = g_ctx.find(d->context);
        if (it == g_ctx.end()) return;
        auto& cd = it->second;

        if (d->callbackSite == CUPTI_API_ENTER) {
            startIfNeeded(cd);
            cd.curKernels++;
            cd.kernelNames.push_back("cuGraphLaunch");
            cout << "[PM-SAMPLING] cuGraphLaunch on device " << cd.deviceId
                 << " (kernel " << cd.curKernels << " of " << cd.maxKernels
                 << " in session " << cd.iterations << ")\n";
        } else if (d->callbackSite == CUPTI_API_EXIT) {
            cuCtxSynchronize();
            if (cd.curKernels >= cd.maxKernels) {
                flushSession(cd, "kernel-rotation");
            }
        }
        return;
    }

    // --- Individual kernel launches (cuLaunchKernel, cuLaunchCooperativeKernel, cuLaunchKernelEx) ---
    if (isKernelLaunchCbid(cbid))
    {
        const CUpti_CallbackData* d = static_cast<const CUpti_CallbackData*>(pCallbackData);
        std::lock_guard<mutex> lk(g_mutex);
        auto it = g_ctx.find(d->context);
        if (it == g_ctx.end()) return; // not profiled / not supported
        auto& cd = it->second;

        // Skip kernels launched during graph capture — they are not executing yet
        if (cd.captureDepth > 0) return;

        if (d->callbackSite == CUPTI_API_ENTER) {
            // start sampling before kernel runs
            if (!cd.samplingActive) startIfNeeded(cd);
            cd.curKernels++;

            // Extract kernel name from callback data
            const char* kernelName = d->symbolName ? d->symbolName :
                                    (d->functionName ? d->functionName : "unknown");
            cd.kernelNames.push_back(kernelName);

            const char* launchType =
                (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel) ? "cuLaunchCooperativeKernel" :
                (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx)          ? "cuLaunchKernelEx" :
                                                                              "cuLaunchKernel";
            cout << "[PM-SAMPLING] " << launchType << " '" << kernelName
                 << "' on device " << cd.deviceId << " (kernel " << cd.curKernels
                 << " of " << cd.maxKernels << " in session " << cd.iterations << ")\n";
        } else if (d->callbackSite == CUPTI_API_EXIT) {
            cuCtxSynchronize(); // block until kernel completes
            // flush after kernel completes
            if (cd.curKernels >= cd.maxKernels) {
                flushSession(cd, "kernel-rotation");
            }
        }
        return;
    }
}

// Register CUPTI callbacks and exit cleanup
static void registerCallbacksOnce()
{
    static bool registered = false;
    if (registered) return;
    registered = true;

    CUpti_SubscriberHandle sub;
    CUPTI_API_CALL(cuptiSubscribe(&sub, (CUpti_CallbackFunc)callback, nullptr));

    // Kernel launch variants
    CUPTI_API_CALL(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API,
                                       CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    CUPTI_API_CALL(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API,
                                       CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel));
    CUPTI_API_CALL(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API,
                                       CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx));

    // CUDA Graph launch
    CUPTI_API_CALL(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API,
                                       CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch));

    // Stream capture tracking (to skip PM sampling during graph capture)
    CUPTI_API_CALL(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API,
                                       CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture));
    CUPTI_API_CALL(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API,
                                       CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2));
    CUPTI_API_CALL(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API,
                                       CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture));

    // Resource tracking
    CUPTI_API_CALL(cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_RESOURCE,
                                       CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
    atexit(atexitHandler);
}

// CUDA will call this when the .so is injected
extern "C" DLLEXPORT int InitializeInjection()
{
    // Register callbacks first; cuptiProfilerInitialize will be called
    // later in initContext() once a CUDA context exists.
    registerCallbacksOnce();

    cout << "[PM-SAMPLING] Injection initialized. "
            "Set INJECTION_METRICS / INJECTION_KERNEL_COUNT / PM_* env vars as needed.\n";
    // From here, callbacks will do the rest.
    return 1;
}