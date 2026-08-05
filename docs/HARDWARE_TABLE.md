# Hardware table - what the tool knows, and how it knows it

Generated from `quantprobe/detect.py` by `weights/make_hardware_table.py`; a smoke
test fails if this file drifts from the code. Bandwidths are THEORETICAL spec peaks
(the law's eta absorbs realism - same convention everywhere in this project).

**Status legend:** `measured` = on this project's reference box, full ladder;
`external` = an independent contributor's scored datapoint (cited);
`spec` = spec-sheet number, no one has validated the law on it yet.

Current census: **1 measured / 3 external / 65 spec-only.** Every `quantprobe bench --contribute` run on a spec-only card is a chance to move a row up - the most valuable submissions are the ones that land OUTSIDE the predicted band.

| card (name match) | vendor | VRAM BW (GB/s, spec) | status | evidence |
|---|---|---|---|---|
| 1060 | NVIDIA | 192 | measured | reference box - eta measured across the full ladder (FINDINGS) |
| rx 5700 xt | AMD | 448 | external | [issue #1 / E-13](https://github.com/FedericoTs/quantprobe/issues/1): predicted 73.1 vs measured 73.18 +/- 0.16 (+0.1%), Vulkan, Windows 11 |
| 3090 | NVIDIA | 936 | external | first external replication (Ryzen 8600G box; source of the channel-count rule) |
| 5060 ti | NVIDIA | 448 | external | E-08: Blackwell-generation replication, +2%..+7.6% inside the published band |
| radeon vii | AMD | 1024 | spec |  |
| rx 7900 xtx | AMD | 960 | spec |  |
| rx 7900 xt | AMD | 800 | spec |  |
| rx 9070 xt | AMD | 640 | spec |  |
| rx 9070 | AMD | 640 | spec |  |
| rx 7800 xt | AMD | 624 | spec |  |
| rx 7900 gre | AMD | 576 | spec |  |
| rx 6950 xt | AMD | 576 | spec |  |
| rx 6900 xt | AMD | 512 | spec |  |
| rx 6800 xt | AMD | 512 | spec |  |
| rx 6800 | AMD | 512 | spec |  |
| vega 64 | AMD | 484 | spec |  |
| rx 5700 | AMD | 448 | spec |  |
| rx 7700 xt | AMD | 432 | spec |  |
| rx 6750 xt | AMD | 432 | spec |  |
| vega 56 | AMD | 410 | spec |  |
| rx 6700 xt | AMD | 384 | spec |  |
| rx 6700 | AMD | 320 | spec |  |
| rx 7600 xt | AMD | 288 | spec |  |
| rx 7600 | AMD | 288 | spec |  |
| rx 5600 xt | AMD | 288 | spec |  |
| rx 6650 xt | AMD | 280 | spec |  |
| rx 6600 xt | AMD | 256 | spec |  |
| rx 6600 | AMD | 224 | spec |  |
| m3 ultra | Apple | 819 | spec | unified memory; estimated eta, unvalidated - bench me |
| m1 ultra | Apple | 800 | spec | unified memory; estimated eta, unvalidated - bench me |
| m2 ultra | Apple | 800 | spec | unified memory; estimated eta, unvalidated - bench me |
| m4 max | Apple | 546 | spec | unified memory; estimated eta, unvalidated - bench me |
| m1 max | Apple | 400 | spec | unified memory; estimated eta, unvalidated - bench me |
| m2 max | Apple | 400 | spec | unified memory; estimated eta, unvalidated - bench me |
| m3 max | Apple | 400 | spec | unified memory; estimated eta, unvalidated - bench me |
| m4 pro | Apple | 273 | spec | unified memory; estimated eta, unvalidated - bench me |
| m1 pro | Apple | 200 | spec | unified memory; estimated eta, unvalidated - bench me |
| m2 pro | Apple | 200 | spec | unified memory; estimated eta, unvalidated - bench me |
| m3 pro | Apple | 150 | spec | unified memory; estimated eta, unvalidated - bench me |
| m4 | Apple | 120 | spec | unified memory; estimated eta, unvalidated - bench me |
| m2 | Apple | 100 | spec | unified memory; estimated eta, unvalidated - bench me |
| m3 | Apple | 100 | spec | unified memory; estimated eta, unvalidated - bench me |
| m1 | Apple | 68 | spec | unified memory; estimated eta, unvalidated - bench me |
| arc a770 | Intel | 560 | spec |  |
| arc a750 | Intel | 512 | spec |  |
| arc b580 | Intel | 456 | spec |  |
| arc b570 | Intel | 380 | spec |  |
| h100 | NVIDIA | 3350 | spec |  |
| a100 | NVIDIA | 1935 | spec |  |
| 5090 | NVIDIA | 1792 | spec |  |
| 4090 | NVIDIA | 1008 | spec |  |
| 5080 | NVIDIA | 960 | spec |  |
| rtx 6000 | NVIDIA | 960 | spec |  |
| 5070 ti | NVIDIA | 896 | spec |  |
| 3080 | NVIDIA | 760 | spec |  |
| 4080 | NVIDIA | 717 | spec |  |
| 5070 | NVIDIA | 672 | spec |  |
| 4070 | NVIDIA | 504 | spec |  |
| 5060 | NVIDIA | 448 | spec |  |
| 3070 | NVIDIA | 448 | spec |  |
| 3060 ti | NVIDIA | 448 | spec |  |
| 2080 | NVIDIA | 448 | spec |  |
| 2070 | NVIDIA | 448 | spec |  |
| 3060 | NVIDIA | 360 | spec |  |
| 2060 | NVIDIA | 336 | spec |  |
| 1080 | NVIDIA | 320 | spec |  |
| 4060 | NVIDIA | 272 | spec |  |
| 1070 | NVIDIA | 256 | spec |  |
| 3050 | NVIDIA | 224 | spec |  |

Missing your card? `quantprobe hw` will name it if the driver registry sees it; pass `--vram-bw` from its spec sheet, run `quantprobe calibrate`, then `bench --contribute` - that is exactly how the first AMD row above got its receipt.
