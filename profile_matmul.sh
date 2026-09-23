#!/bin/bash
# profile all three matmul kernels with ncu and dump the sections that
# actually matter for the naive -> tiled -> regtile story
#
# needs gpu performance counter access (ERR_NVGPUCTRPERM otherwise):
#   - root: echo "$USER ALL=(root) NOPASSWD: /usr/local/cuda/bin/ncu" | sudo tee /etc/sudoers.d/ncu
#   - WSL: the windows driver gates counters too, all three of these
#     must be 0 in the registry, then reboot:
#       Services\nvlddmkm
#       Services\nvlddmkm\Global\NVTweak
#       Services\nvlddmkm\Parameters\Global\NVTweak
#   - if the NVIDIA Control Panel has Developer -> Manage GPU Performance
#     Counter Policy, that switch does the same thing in one click
# on my laptop none of that was enough (OEM driver), so the script falls
# back to the event-timing + roofline path -- see PROFILING.md
#
# usage: ./profile_matmul.sh

set -e
cd "$(dirname "$0")"

NCU=${NCU:-/usr/local/cuda/bin/ncu}
APP=./matmul_profile

# probe once whether counters are usable before doing real work.
# note ncu exits 0 even when counters are blocked, so check the output,
# not the exit code
if ! command -v $NCU >/dev/null 2>&1; then
    echo "nsight-compute not found, falling back to event timings"
    FALLBACK=1
else
    PROBE=$(sudo -n $NCU -k regex:naive --launch-count 1 \
                --section SpeedOfLight $APP 2>&1 || true)
    if echo "$PROBE" | grep -q ERR_NVGPUCTRPERM; then
        echo ""
        echo "!! ERR_NVGPUCTRPERM: counter access blocked on this machine"
        echo "!! (see the header of this script + PROFILING.md for the checklist)"
        echo "!! falling back to event timings + roofline anchors"
        echo ""
        FALLBACK=1
    fi
fi

if [ "${FALLBACK:-0}" = "1" ]; then
    make -s roofline matmul_profile
    ./roofline
    for n in 512 1024 2048 4096; do
        echo "--- N=$n"
        ./$APP $n | head -3
    done
    exit 0
fi

SECTIONS="SpeedOfLight,Occupancy,MemoryWorkloadAnalysis,SchedulerStats,WarpStateStats,InstructionStats"

for k in naive tiled regtile; do
    echo "=== $k ==="
    # -c 1: profile just the first (warmup) launch per kernel
    sudo -n $NCU -k regex:$k --launch-count 1 \
        --section SpeedOfLight \
        --section Occupancy \
        --section MemoryWorkloadAnalysis \
        --section WarpStateStats \
        -o $k -f $APP 2>&1 | grep -E "Duration|Compute \(SM\)|Memory Thr|Achieved Occupancy|Stall|L2 Hit|Mem Busy|Max Bandwidth" || true
    echo
done

for k in naive tiled regtile; do
    $NCU --import $k.ncu-rep --page details > $k.txt 2>/dev/null || true
done

echo "done. reports: naive/tiled/regtile .ncu-rep + .txt"
