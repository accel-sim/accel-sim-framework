 /***************************************************************************\
|*                                                                           *|
|*                                                                           *|
|*      Copyright 2010-2012 NVIDIA Corporation.  All rights reserved.        *|
|*                                                                           *|
|*   NOTICE TO USER:                                                         *|
|*                                                                           *|
|*   This source code is subject to NVIDIA ownership rights under U.S.       *|
|*   and international Copyright laws.  Users and possessors of this         *|
|*   source code are hereby granted a nonexclusive, royalty-free             *|
|*   license to use this code in individual and commercial software.         *|
|*                                                                           *|
|*   NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE     *|
|*   CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR         *|
|*   IMPLIED WARRANTY OF ANY KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH      *|
|*   REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF         *|
|*   MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR          *|
|*   PURPOSE. IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL,            *|
|*   INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES          *|
|*   WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN      *|
|*   AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING     *|
|*   OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE      *|
|*   CODE.                                                                   *|
|*                                                                           *|
|*   U.S. Government End Users. This source code is a "commercial item"      *|
|*   as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting       *|
|*   of "commercial computer  software" and "commercial computer software    *|
|*   documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)   *|
|*   and is provided to the U.S. Government only as a commercial end item.   *|
|*   Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through        *|
|*   227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the       *|
|*   source code with only those rights set forth herein.                    *|
|*                                                                           *|
|*   Any use of this source code in individual and commercial software must  *|
|*   include, in the user documentation and internal comments to the code,   *|
|*   the above Disclaimer and U.S. Government End Users Notice.              *|
|*                                                                           *|
|*                                                                           *|
 \***************************************************************************/
/********************************************************************
 *      Modified by:
 * Tayler Hetherington, University of British Columbia
 * Vijay Kandiah, Northwestern University
 ********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <signal.h>
#include <nvml.h>

// CTRL+C handler
static volatile int exitFlag = 0;
void intHandler(int v)
{
    exitFlag = 1;
}

int init()
{
    printf("Initializing NVML... ");
    nvmlReturn_t res;
    res = nvmlInit();
    if( res != NVML_SUCCESS ) {
        printf("Error: failed to initialize NVML: %s\n", nvmlErrorString(res));
        return 0;
    }
    printf("Success\n");
    return 1;
}

int shutdown()
{
    printf("Shutting down NVML... ");
    nvmlReturn_t res;
    res = nvmlShutdown();
    if( res != NVML_SUCCESS ) {
        printf("Error: Failed to shutdown NVML: %s\n", nvmlErrorString(res));
        return 0;
    }
    printf("Success\n");
    return 1;
}

int getDevice(int devId, nvmlDevice_t* dev, nvmlUnit_t* unit)
{
    nvmlReturn_t res;
    unsigned numDev = 0;

    printf("Getting device... ");

    res = nvmlDeviceGetCount(&numDev);
    if( res != NVML_SUCCESS ) {
        printf("Error: Failed to get number of devices: %s\n", nvmlErrorString(res));
        return 0;
    }

    if( devId >= numDev ) {
        printf("Error: Invalid device ID: %d\n", devId);
        return 0;
    }

    char devName[NVML_DEVICE_NAME_BUFFER_SIZE];

    res = nvmlDeviceGetHandleByIndex(devId, dev);
    if( res != NVML_SUCCESS ) {
        printf("Error: failed to get handle for device %i: %s\n", devId, nvmlErrorString(res));
        return 0;
    }

    res = nvmlDeviceGetName(*dev, devName, NVML_DEVICE_NAME_BUFFER_SIZE);
    if( res != NVML_SUCCESS ) {
        printf("Error: failed to get name of device %i: %s\n", devId, nvmlErrorString(res));
        return 0;
    }

    printf("Selected device %d: %s\n", devId, devName);

#if 0
    printf("Getting NVML Unit... ");
    unsigned unitCount = 0;
    nvmlUnitGetCount(&unitCount);
    printf("%d units\n", unitCount);
    res = nvmlUnitGetHandleByIndex( devId, unit );
    if( res != NVML_SUCCESS ) {
        printf("Error: failed to get unit %i: %s\n", devId, nvmlErrorString(res));
        return 0;
    }
#endif

    return 1;
}

int processIsAlive(int pid)
{
	char command[256];
	sprintf(command, "ps | grep -c %d", pid);
	FILE* f;
	f = popen(command, "r");
	if (f == NULL) {
		printf("Failed to run command\n");
		pclose(f);
		return 0;
	}
	char line[8];
	if (fgets(line, sizeof(line), f) != NULL) {
		pclose(f);
		return atoi(line);
	}
	pclose(f);
	return 0;
}

int measurePower(char* oFileName, int csv, int devId, nvmlDevice_t* dev, int sampleRate, int numSamples, nvmlUnit_t* unit, int printPsuInfo, int pid, int temp_cutoff_T)
{
    nvmlReturn_t res;
    unsigned int mWatts = 0;
	unsigned int temperature = 0;
    nvmlPSUInfo_t psu;
    nvmlPstates_t pState;
    int samplesRemaining = (numSamples == -1 ? 1 : numSamples);
    double avgWatts = 0.0;
    unsigned int sampleCount = 0;
    FILE *f;
    nvmlUtilization_t util;
    bool temp_cutoff = false;
    if( oFileName ) {
        f = fopen(oFileName, "w");
        if( f == NULL ) {
            printf("Error: failed to open file %s\n", oFileName);
            return 0;
        }
    }

    printf("\n\nStarting power measurement...\n");

    while( samplesRemaining > 0 ) {
        // Throttle to sample rate
        usleep(sampleRate * 1000.0);
        //increment number of samples processed


		if (pid != 0 && processIsAlive(pid) == 0) {
			printf("Application terminated. Closing profiler...\n");
			break;
		}
        res = nvmlDeviceGetUtilizationRates ( *dev, &util);

        if(util.gpu < 95) {
            if (samplesRemaining >1){
                samplesRemaining--;
                continue;
            }
            else if( numSamples == -1 )
				continue;
			else
                break;
        }



		if (temp_cutoff_T)
		{
			res = nvmlDeviceGetTemperature( *dev, NVML_TEMPERATURE_GPU, &temperature );
			if( res != NVML_SUCCESS ) {
				printf("Error: failed to get temperature for device %i: %s\n", devId, nvmlErrorString(res));
				return 0;
			}
			if (temperature == temp_cutoff_T) {
                temp_cutoff = true;
				printf("Cutoff temperature %d C reached: concluding power measurements\n", temp_cutoff_T);
				samplesRemaining = 1; // record the temperature one last time
                avgWatts = 0.0;
                sampleCount = 0;
			}
		}
        res = nvmlDeviceGetPowerUsage( *dev, &mWatts );
        if( res != NVML_SUCCESS ) {
            printf("Error: failed to get power for device %i: %s\n", devId, nvmlErrorString(res));
            return 0;
        }

        if( printPsuInfo ) {
            res = nvmlDeviceGetPerformanceState(*dev, &pState);
            //res = nvmlUnitGetPsuInfo(*unit, &psu);
            if( res != NVML_SUCCESS ) {
                printf("Error: failed to get pState info for device %i: %s\n", devId, nvmlErrorString(res));
                return 0;
            }
        }

        double watts = (double)mWatts / 1000.0;
        avgWatts += watts;
        sampleCount++;

        if(( numSamples != -1 ) || (temp_cutoff == true))
            samplesRemaining--;

        if( exitFlag ) {
            printf("Stopping power measurement...\n");
            break;
         }
    }

    if (sampleCount == 0)
        avgWatts = 0.0;
    else
        avgWatts /= sampleCount;
    if( csv ) {
        if( oFileName ) {
            if( !printPsuInfo )
                fprintf(f, "%.4lf, ", avgWatts);
            else
                fprintf(f, "%.4lf, %d\n", avgWatts, pState);
        } else {
            if( !printPsuInfo )
                printf("%.4lf, ", avgWatts);
            else
                printf("%.4lf, %d\n", avgWatts, pState);
        }
    } else {
        if( oFileName ) {
            fprintf(f, "Power draw = %.4lf W \n", avgWatts);
        } else {
            if( !printPsuInfo )
                printf("Power draw = %.4lf W\n", avgWatts);
            else
                printf("Power draw = %.4lf W, power state = %d \n", avgWatts, pState);
        }
    }


    if ((temp_cutoff_T) && (!temp_cutoff))
        printf("WARNING: TEMPERATURE CUTTOFF NOT REACHED \n\n");
    printf("\n\n");
    return 1;
}

void printHelp()
{
    printf("\n=============================================================================================\n");
    printf("-h:\t\t\t\tPrints this help message\n");
    printf("-o <output file name>:\t\tOutput file\n");
    printf("-c:\t\t\t\tWrite csv format\n");
    printf("-d <device number>: \t\tNVIDIA device to collect power\n");
    printf("-r <sample rate>: \t\tSample rate to collect power in milliseconds\n");
    printf("-n <number of samples>: \tNumber of power samples to collect (-1 == poll until CTRL+C)\n");
    printf("-p: \t\t\t\tPrint PSU info as well \n");
	printf("-a: <process name>: \t\tApplication to profile with this profiler. Profiler will stop when process terminates\n");
	printf("-t: <temp>:\t\tCutoff temperature in degrees C. Profiler will stop profiling if the GPU reaches this temperature\n");

    printf("\nCTRL+C will stop the power measurements and shutdown NVML\n");
    printf("=============================================================================================\n\n");
}

int parseOptions(int argc, char** argv, char** outFileName, int* csv, int* devId, int* sampleRate, int* numSamples, int* printPsuInfo, char** pname, int* temp_cutoff_T)
{
    int opt;
    opterr = 0;
    while( (opt = getopt(argc, argv, "ho:cd:r:n:pa:t:")) != -1 ) {
        switch(opt) {
            case 'h':
                printHelp();
                return 2;

            case 'o':
                *outFileName = optarg;
                break;

            case 'c':
                *csv = 1;
                break;

            case 'd':
                *devId = atoi(optarg);
                break;

            case 'r':
                *sampleRate = atoi(optarg);
                break;

            case 'n':
                *numSamples = atoi(optarg);
                break;

            case 'p':
                *printPsuInfo = 1;
                break;

			case 'a':
				*pname = optarg;
				break;

			case 't':
				*temp_cutoff_T = atoi(optarg);
				break;

            default:
                printf("Error: unknown option\n");
                return 0;
        }
    }

    printf("Options: \n"
            "\tOutput File = %s\n"
            "\tCSV mode = %d\n"
            "\tDevice ID = %d\n"
            "\tSample Rate = %d ms\n"
            "\tNumber of power samples = %d\n"
            "\tPrint PSU info = %d\n"
			"\tProcess Name = %s\n\n"
			"\tTemperature Cutoff (0 iff disabled) = %d\n",
            *outFileName, *csv, *devId, *sampleRate, *numSamples, *printPsuInfo, *pname, *temp_cutoff_T);

    return 1;
}

int main(int argc, char** argv)
{
    nvmlReturn_t res;
    nvmlDevice_t dev;
    nvmlUnit_t unit;
    int ret = 0;

    // Defaults
    int devId = 0;
    int sampleRate = 100;
    int numSamples = -1;
    char* oFileName = NULL;
    int csv = 0;
    int printPsuInfo = 0;
	char* pname = NULL;
	int pid = 0;
	int temp_cutoff_T = 0;

    // Setup CTRL+C handler
    signal(SIGINT, intHandler);

    ret = parseOptions(argc, argv, &oFileName, &csv, &devId, &sampleRate, &numSamples, &printPsuInfo, &pname, &temp_cutoff_T);
    if( !ret )
        return EXIT_FAILURE;

    if( ret == 2)
        return EXIT_SUCCESS;

    if( !init() )
        return EXIT_FAILURE;

    if( !getDevice(devId, &dev, &unit) ) {
        shutdown();
        return EXIT_FAILURE;
    }

	if (pname != NULL) {
		char command[256];
		FILE* f;
		char pname_short[16];
		int plength = strlen(pname) < 15 ? strlen(pname) : 15;
		for (int i = 0; i <= plength; i++)
		{
			if (i != plength)
				pname_short[i] = pname[i];
			else
				pname_short[i] = '\0';
		}
		sprintf(command, "ps | grep '%s' | awk '{print $1}'", pname_short);
		printf("Attempting to get pid\n");
		f = popen(command, "r");
		if( f == NULL ) {
			printf("Error: failed to run ps command.\n");
			pclose(f);
			shutdown();
			return EXIT_FAILURE;
		}
		char line[8];
		if (fgets(line, sizeof(line), f)) {
			pid = atoi(line);
		} else {
			pclose(f);
			printf("Error: application not found in process list. Make sure to start application before profiling!\n");
			shutdown();
			return EXIT_FAILURE;
		}
		pclose(f);
		printf("Got the pid\n");
	}

    if( !measurePower(oFileName, csv, devId, &dev, sampleRate, numSamples, &unit, printPsuInfo, pid, temp_cutoff_T) ) {
        shutdown();
        return EXIT_FAILURE;
    }

    shutdown();

    printf("Complete \n");

    return EXIT_SUCCESS;

}
