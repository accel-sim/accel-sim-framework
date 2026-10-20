# Accel-Sim pre-collected traces

SASS traces for [Accel-Sim](https://github.com/accel-sim/accel-sim-framework), one set per
GPU / workload / argument set, in the standard Accel-Sim layout:

```
<GPU>/traces/device-0/<cuda>/<workload>/<args>/traces/   kernelslist.g + kernel-*.tracez
<GPU>/device-0/<cuda>/<workload>/<args>/                  Nsight Compute stats (ncu.csv.*)
```

The Nsight Compute rows are in the same order as the kernels in `kernelslist.g`. A few
kernels that do not complete in simulation are left out of both.

Every set is one tar, split into parts smaller than 2 GB
(`<GPU>__<workload>__<args>.tar.partNNN`). `manifest.json` lists every part with its size
and SHA-256, plus the SHA-256 of the whole tar.

## Download with the script (recommended)

From the root of an Accel-Sim checkout:

```bash
./get-accel-sim-traces.py -l                                    # list the sets in this release
./get-accel-sim-traces.py -a H200/inference_bert.sh/32_256      # one argument set
./get-accel-sim-traces.py -a H200/inference_sdxl.sh             # every argument set of a workload
./get-accel-sim-traces.py -a H200/all                           # everything for one GPU
```

The script checks every part and the whole tar against `manifest.json` and extracts under
`./hw_run/`. Use `-d <dir>` for another location, `--keep_parts` to keep the downloaded
parts, and `--repo` / `--tag` to read a different release.

Then simulate and correlate:

```bash
./util/job_launching/run_simulations.py -B bert_inference -C H200-SASS -T ./hw_run/H200/traces/device-0/12.8 -N bert
./util/job_launching/get_stats.py -R -k -K -N bert | tee bert.stats.csv
./util/plotting/plot-correlation.py -c bert.stats.csv -H ./hw_run/H200
```

## Download by hand

```bash
SET=H200__inference_bert.sh__32_256.tar
# download every $SET.partNNN asset of this release, then:
sha256sum $SET.part*            # compare with "parts" in manifest.json
cat $SET.part* > $SET           # the parts are plain byte splits, concatenated in order
sha256sum $SET                  # compare with "sha256" of the set in manifest.json
tar -xf $SET -C ./hw_run
```

## Available sets
