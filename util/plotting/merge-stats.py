#!/usr/bin/env python3

from __future__ import print_function
from optparse import OptionParser
import os

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

import sys

sys.path.insert(0, os.path.join(this_directory, "..", "job_launching"))
import common

import csv

BUILD_STATS = ["Accel-Sim-build", "GPGPU-Sim-build"]
BUILD_KERNEL = "all_kernels"


def split_column(appargs_kname):
    """'app/args--kernel--N' -> ('app/args', 'kernel--N'); None for AVG columns."""
    first_delimiter = appargs_kname.find("--")
    if first_delimiter < 0:
        return None
    appargs = appargs_kname[:first_delimiter]
    # "AV--VG": the AVG column of a get_stats -A file, as older merges stored it
    if appargs == "AV":
        return None
    return appargs, appargs_kname[first_delimiter + 2 :]


def build_suffix(config, values):
    """The -accel-<sha>-gpgpu-<sha> name of a get_stats config row, or None."""
    accel = set(v[16:23] for v in values["Accel-Sim-build"] if v != "NA")
    gpgpu = set(v[21:28] for v in values["GPGPU-Sim-build"] if v != "NA")
    if not accel or not gpgpu:
        return None
    if len(accel) > 1 or len(gpgpu) > 1:
        exit(
            "Config {0} has more than one build: accel {1}, gpgpu {2} - one get_stats row "
            "must come from one build".format(config, accel, gpgpu)
        )
    return "-accel-{0}-gpgpu-{1}".format(accel.pop(), gpgpu.pop())


def get_csv_data_for_merge(filepath):
    """Parse a configs-as-rows get_stats/merge-stats CSV.

    Returns (stats, columns, values): the stat names in file order, the
    (appargs, kname) columns of each stat, and {stat: {config: [values]}}.
    """
    stats = []
    columns = {}
    values = {}
    with open(filepath, "r") as data_file:
        state = "start"
        for row in csv.reader(data_file):
            if len(row) != 0 and row[0].startswith("----"):
                state = "find-stat"
                continue
            if state == "find-stat":
                current_stat = row[0]
                stats.append(current_stat)
                values[current_stat] = {}
                state = "find-apps"
                continue
            if state == "find-apps":
                header = [split_column(c) for c in row[1:]]
                if current_stat in BUILD_STATS:
                    # older merges printed the build stats under kernel columns, all NA
                    header = [c if c and c[1] == BUILD_KERNEL else None for c in header]
                columns[current_stat] = [c for c in header if c is not None]
                state = "process-cfgs"
                continue
            if state == "process-cfgs":
                if len(row) == 0:
                    state = "start"
                    continue
                values[current_stat][row[0]] = [
                    v for c, v in zip(header, row[1:]) if c is not None
                ]
    return stats, columns, values


def load_file(filepath):
    """Key one file's data as stat_map[kname + appargs + config + stat].

    A config row straight from get_stats is renamed <config>-accel-<sha>-gpgpu-<sha>
    in every stat, the build stats included; rows of a merged file already are.
    """
    stats, columns, values = get_csv_data_for_merge(filepath)
    for stat in BUILD_STATS:
        if stat not in values:
            values[stat] = {}
            columns[stat] = []
    configs = []
    for stat in stats:
        for config in values[stat]:
            if config not in configs:
                configs.append(config)
    rename = {}
    for config in configs:
        per_build = {
            stat: values[stat].get(config, []) for stat in BUILD_STATS
        }
        suffix = build_suffix(config, per_build)
        if suffix is None or config.endswith(suffix):
            rename[config] = config
        else:
            rename[config] = config + suffix
    stat_map = {}
    for stat in stats:
        for config, row in values[stat].items():
            if len(row) != len(columns[stat]):
                exit(
                    "{0}: stat {1} config {2} has {3} values for {4} columns".format(
                        filepath, stat, config, len(row), len(columns[stat])
                    )
                )
            for (appargs, kname), v in zip(columns[stat], row):
                if v != "NA":
                    stat_map[kname + appargs + rename[config] + stat] = v
    return stats, columns, [rename[c] for c in configs], stat_map


def ordered_add(lst, seen, item):
    if item not in seen:
        seen.add(item)
        lst.append(item)


parser = OptionParser()
parser.add_option(
    "-c", "--csv_files", dest="csv_files", help="Files to merge", default=""
)
parser.add_option(
    "-R",
    "--configs_as_rows",
    dest="configs_as_rows",
    help="When printing merged files, are configs as rows?",
    action="store_true",
)
(options, args) = parser.parse_args()

csv_files = []
for csvf in options.csv_files.split(","):
    try:
        csv_files.append(common.file_option_test(csvf, "", this_directory))
    except common.PathMissing as e:
        print("Warning path {0}. Continuing".format(e), file=sys.stderr)

# Union of every input: an app, kernel, stat or config any file has is kept,
# and a value in a later file replaces the same one in an earlier file.
new_stats = {}
new_configs, seen_configs = [], set()
all_stats, seen_stats = list(BUILD_STATS), set(BUILD_STATS)
apps_and_args, seen_apps = [], set()
all_named_kernels, seen_kernels = {}, {}
per_file = []
for csvf in csv_files:
    if os.path.getsize(csvf) == 0:
        print("Skipping empty {0}".format(csvf), file=sys.stderr)
        continue
    print("Processing {0}".format(csvf), file=sys.stderr)
    stats, columns, configs, stat_map = load_file(csvf)
    per_file.append((csvf, stats, columns, configs, stat_map))
    new_stats.update(stat_map)
    for config in configs:
        if config in seen_configs:
            print(
                'Found redundant config: {0} in csvf: "{1}" - later values win.'.format(
                    config, csvf
                ),
                file=sys.stderr,
            )
        ordered_add(new_configs, seen_configs, config)
    for stat in stats:
        ordered_add(all_stats, seen_stats, stat)
        # an app that failed in every config has build columns and no kernels
        for appargs, kname in columns[stat]:
            if appargs not in seen_apps:
                all_named_kernels[appargs], seen_kernels[appargs] = [], set()
            ordered_add(apps_and_args, seen_apps, appargs)
            if stat not in BUILD_STATS:
                ordered_add(all_named_kernels[appargs], seen_kernels[appargs], kname)

build_kernels = {appargs: [BUILD_KERNEL] for appargs in apps_and_args}

# Coverage: everything an input has must be printed
for csvf, stats, columns, configs, stat_map in per_file:
    for stat in stats:
        kernels = build_kernels if stat in BUILD_STATS else all_named_kernels
        missing = [
            "{0}--{1}".format(a, k)
            for a, k in columns[stat]
            if a not in kernels or k not in kernels[a]
        ]
        if missing:
            exit(
                "merge-stats coverage: {0} stat {1} has columns the output lacks: {2}".format(
                    csvf, stat, missing[:5]
                )
            )
    if not set(configs) <= seen_configs or not set(stats) <= seen_stats:
        exit("merge-stats coverage: {0} configs or stats missing in output".format(csvf))

for stat in all_stats:
    common.print_stat(
        stat,
        build_kernels if stat in BUILD_STATS else all_named_kernels,
        apps_and_args,
        new_configs,
        new_stats,
        options.configs_as_rows,
        False,
    )
