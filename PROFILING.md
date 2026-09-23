# profiling the matmul kernels

goal: figure out WHY naive (07) is slow, what tiled (08) fixes, and what
regtile (20/23) still leaves on the table. TL;DR at the bottom.

## the ncu detour (or: how I spent an evening fighting a driver)

plan was to profile `./matmul_profile` with Nsight Compute and read the
Speed-of-Light numbers. on this laptop (RTX 3060 Laptop, WSL2, driver
610.62 via /dev/dxg) ncu refuses with `ERR_NVGPUCTRPERM` no matter what.
notes for whoever hits the same wall:

1. ncu needs root here -> `sudoers.d` rule for the ncu binary
2. WSL2 counters are gated by the WINDOWS driver, not the linux side.
   there are THREE registry locations with `RmProfilingAdminOnly`, all
   need to be 0, and a reboot is needed:
   - `Services\nvlddmkm`
   - `Services\nvlddmkm\Global\NVTweak`
   - `Services\nvlddmkm\Parameters\Global\NVTweak`   <- the sneaky one
3. after all that: still blocked. the NVIDIA Control Panel
   "Developer -> Manage GPU Performance Counter Policy" switch is the
   documented way, but this OEM driver setup doesn't expose it.

so: counters are off the table on this machine. nsys 2024.5 does trace
CUDA API calls fine (18 launches, malloc costs visible) but GPU kernel
records come back empty too (CUPTI/driver version mismatch on the new
dxg stack). the analysis below is therefore built from cuda-event
timings + measured hardware ceilings instead of counter readouts. the
ncu commands to re-run on an unrestricted machine are at the end.

## measured ceilings (24_roofline_anchor.cu, best of 3-5 runs)

| anchor | measured | note |
|---|---|---|
| D2D copy bandwidth | 225 GB/s | 1 GB memcpy, counts read+write |
| streaming (saxpy)  | 286 GB/s | 2 reads + 1 write per element |
| sgemm 4096, cuBLAS | 7.14 TFLOP/s | practical FP32 compute roof |

spec sheet says ~360 GB/s DRAM and ~13 TFLOP/s FP32 for this GPU;
measured 80% / ~55% is normal for a laptop 95W config. use the MEASURED
numbers as the roof, not the marketing ones.

## the kernels, swept over N (matmul_profile, best of 5 launches)

| N | naive | tiled | regtile |
|---|---|---|---|
| 512 | 0.65 TF/s | 0.83 TF/s | 2.98 TF/s |
| 1024 | 0.52 | 0.68 | 3.25 |
| 2048 | 0.61 | 1.02 | 4.15 |
| 4096 | 0.73 | 0.94 | 4.50 |

two things jump out:

1. naive is FLAT (~0.5-0.7 TF/s) from 512 to 4096. problem size doesn't
   matter at all -> the limit is not DRAM (traffic scales with N) but
   something fixed per thread. that something is the B-column access:
   threads of a warp read B[k*n + c] with stride n*4B, so one warp
   request turns into 32 separate 32B sectors instead of 4 coalesced
   ones. 32x the L1TEX wavefronts, 32x the latency exposure, and the
   FMA pipes starve. sanity check: if naive were DRAM-bound at 286 GB/s
   with its worst-case 2*N^3*4B of traffic, N=4096 would take 1.9 s.
   it does 0.188 s -> caches absorb the traffic, the cost is in
   wavefronts + latency, which is exactly what tiling attacks.
2. tiled improves 1.3-2x but plateaus around 1 TF/s. each thread does
   1 FMA per k-step with 2 shared-memory loads (As[y][k], Bs[k][x]).
   that's 8 bytes of shared traffic per FMA -> shared/issue bound, the
   known 16x16-tile 1-output-per-thread plateau.

regtile fixes the second one: 4x4 outputs per thread means 16 FMAs per
8 shared loads, i.e. 4x the arithmetic per byte of shared traffic.
4.5 TF/s at N=4096 = 63% of the cuBLAS roof measured on the same card.
the remaining ~40%: 32KB shared/block caps occupancy at ~1-2 blocks per
SM (30 SMs), no double buffering / async copies, and per-thread serial
k-loops that leave some latency exposed. all fixable, none trivial.

## conclusions (what I'd say in an interview)

- naive: bound by L1TEX/memory pipeline (uncoalesced column access,
  32 sectors per warp request), NOT by DRAM and NOT by FLOPs.
- tiled: coalescing fixed -> now bound by shared-memory throughput
  (8 B shared per FMA).
- regtile: arithmetic intensity per shared-byte raised 4x -> 63% of
  the practical compute roof; rest is occupancy + buffering.
- cuBLAS 7.1 TF/s is the roof to beat on this card; nobody hand-writes
  GEMM for production, but walking this ladder is how you learn what
  the library is doing for you.

## reproducing

```bash
make matmul_profile roofline
./roofline                       # ceilings
./matmul_profile 1024            # sweep: 256..4096
./profile_matmul.sh              # ncu collection (needs counter access)
```

on a machine where counters work, per kernel:

```bash
ncu -k regex:naive --launch-count 1 --set full ./matmul_profile
```

numbers to look at, and what I expect them to show:
- Speed of Light: naive -> SM% low, mem% high (memory pipeline busy,
  FMA pipes idle); regtile -> SM% high
- Memory workload: naive L1TEX sectors/request ~32 vs tiled ~4
- Warp stalls: naive -> long scoreboard (global latency);
  tiled -> short scoreboard / MIO (shared);
  regtile -> math pipe throttled
- Occupancy: regtile limited by shared memory per block

if any of these DON'T match the story above, the story is wrong and the
profiler is right -- that's the whole point of profiling.
