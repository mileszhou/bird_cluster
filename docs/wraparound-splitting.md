## Methodology (2018/2019 entries)

The 2018 and 2019 splits below were derived automatically, not hand-inspected like
the earlier years, since those two years have no jpg/ export yet to eyeball. Tool:
`scripts/analyze_wraparound.py data/Photos-YY` (safe to re-run any time; read-only).

1. Parse each xmp filename into `(camera_code, counter)` — regex strips an optional
   `YYYYMMDD-` date-disambiguation prefix (added on import when Lightroom/macOS saw a
   name clash), an optional `-N` virtual-copy suffix, and an optional
   `_hostname_timestamp_Conflict` sync-conflict suffix first.
2. Group xmp files by trip folder, sorted chronologically by the folder's `YYYY-MM[-DD]`
   name prefix.
3. Per trip, collect the set of exact `(code, counter)` pairs present, and separately
   the per-code `[min, max]` counter range.
4. Report every `(code, counter)` pair that repeats across trips as a direct collision
   (raw evidence, independent of any split).
5. Greedy sweep in date order: keep the current segment open and merge each trip in
   until the next trip would collide with something already in it, then cut. This is
   provably the fewest possible segments *for whatever collision test is used*, since
   the "no collision in a block" constraint only gets easier to satisfy as you remove
   trips — any other valid partition must cut at or before greedy's cut point.

Two collision tests were run, giving two segment counts:

| | exact-match (used below) | range-overlap (stricter) |
|---|---|---|
| 2018 | 7 segments | 20 segments |
| 2019 | 18 segments | 27 segments |

**Exact-match** only flags a collision when a specific surviving xmp frame's number
repeats in a later trip — optimistic, since it says nothing about frames that were
shot but never made it into this library (deleted, never touched in Lightroom, etc.).
**Range-overlap** instead cuts whenever two trips' per-code counter ranges overlap at
all, whether or not a specific number actually repeats — safer, at the cost of ~3x
more segments. The entries below use the exact-match (coarser) split; re-run the
script with the numbers above in mind if the finer, safer boundaries are wanted for
the actual re-export.

### Year 2025
    1. 01/01 -- 05/04
    2. 05/05 -- 07/06.1
    3. 07/06.2 -- 07/17 
    4. 07/18 -- end        DSC8285 - 

### Year 2024
    1.  01/01 - 01/09   8533 - 7065
        01/10 - 04/18   7238 - 6813
        04/19 - 04/28   6848 - 6798
        04/30 - 05/14   6873 - 5437
        05/15 - 11/02   5437+ - ... 850: 2806 - 2452
        11/03 - 11/12   2512 - 2336
        11/12 - 11/24   2349 - 1003
        11/25 - 12/25

### Year 2023
    01/01 - 10/04   6417 - 5948
    10/05 - 12/27   5595 - 

### Year 2022
    No wraparound

### Year 2021
    0113 - 0224 8993 - 
    0225 - 0415am 8082:
    0415pm - 0505 7638 - 
    0507 - 0924     6826
    0926 - 1015     6111: 
    1017 - 1109     5269:
    1112 - 1219     5189:

### Year 2019
    (dense back-to-back trips force many small segments, esp. the Feb 山公园 multi-pond
     outing and the Aug-Sep Europe trip. Derived from XMP filenames via
     `python3 scripts/analyze_wraparound.py data/Photos-19`, which also lists every
     colliding (camera-code, counter) pair and the per-camera-code counter ranges.)
    1.  01/13 -- 02/02   (山公园 -> 山公园77号塘)
    2.  02/03 -- 02/04a  (山公园4号塘 -> 35号塘)
    3.  02/04b           (山公园32号塘, single trip)
    4.  02/04c -- 02/05  (山公园35号塘 -> 山公园11号塘&棕背田鸡)
    5.  02/06 -- 02/07a  (旧街&37号塘 -> 02/07 untitled)
    6.  02/07b -- 02/08a (黑颈长尾稚 -> 02/08 untitled)
    7.  02/08b -- 02/12a (盈江2,7,8,10,5号塘 -> 盈江11号塘&猛隼)
    8.  02/12b           (那邦&腾冲, single trip)
    9.  03/04 -- 04/13a  (逸城 -> Telescopes)
    10. 04/13b -- 06/08a (t -> 九山公园寿带)
    11. 06/08b -- 08/31a (山公园 -> Brussells)
    12. 08/31b -- 09/05  (Vondelpark Amsterdam -> Versaille Palace)
    13. 09/06 -- 09/07a  (Fontainblue Palace -> Musee d'Orsey)
    14. 09/07b -- 10/28  (Back to London -> Wuxi Zoo)
    15. 10/29 -- 11/16   (无锡贡湖湾 -> 顾剑萍)
    16. 11/17 -- 11/22   (镰仓 -> Venice)
    17. 11/23            (Florance, single trip)
    18. 11/26 -- 12/07   (Rome -> 红翡翠)

### Year 2018
    (multiple camera bodies active simultaneously (D5C/D5D/D5S/D5V/D8S/DFS/DSC), each
     wrapping its counter independently; segments below are collision-free across all
     bodies combined. Derived from XMP filenames via
     `python3 scripts/analyze_wraparound.py data/Photos-18`.)
    1. 01/01 -- 02/21   (中华秋沙鸭 -> 山公园35号塘)
    2. 02/22 -- 03/18   (山公园32号塘 -> 酒窖)
    3. 03/24 -- 05/10   (世纪公园 -> Beijing Zoo)
    4. 05/12 -- 08/05a  (乾隆生态园 -> 永禾村)
    5. 08/05b -- 10/20  (安禾村 -> 平湖-昵称回门)
    6. 10/21 -- 11/01   (Shanghai-Emily -> 南汇嘴&上虞)
    7. 11/03 -- 12/31   (南汇嘴&上虞 -> 大河饭店)