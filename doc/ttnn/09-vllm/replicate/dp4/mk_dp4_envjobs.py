"""The Galaxy TP8 x DP4 env jobs + targets (operator delivery, program 2026-10-28T12:20:55Z item (2); form switch 12:25:35Z):
four standing TP8 servers, one per 2x4 block, on the served form of record:
  --form t6d (default, program 12:25:35Z: the DP4 form switches to k4d8t6d on its GREEN mesh gate) = k4d8t6d: tree
    tt-metal-s9cfk4d8t6d 5aec76ac6c7, the k4d8t6 env + GLM53_PMK_DSA_POST=1 (PAIRFWD p15, WS_O 0, RTAIL_BATCH 1, FAST_SCAN 1);
  --form f2 (the fallback on a k4d8t6d RED) = k4d8t5wp6f2: tree tt-metal-s9cfk4d8t5w dbfa7ac6290;
both on plugin srv/s9hp5bc 5bcf424 + AUDIT=1 + COLLECT_SERIAL=0 (the five export keys unset), no R1b, no MTP, no DFlash2.
Per block N (tag <form tag>dp4b<N>), from the Galaxy form of the form's env job (mk_glx_envjob.py: GRID_X 11, the 1530 pin):
  - a HYBRID-role (H) overlay env job work-s9/envjobs/int27<tag>dp4b<N>pdH_img_env.done (pd/glx_pd_role.sh with GLX_PD_ROLE=H
    GLX_PD_TAG=<tag>dp4b<N> reads it; glx_image_mock.sh GLX_TAG=<tag>dp4b<N> GLX_PD_ROLE=H mocks it);
  - a target scripts/targets/int27<tag>dp4b<N>_resident.sh = the form's served target with the tray's identity, env job and
    per-block dirs, the strict self-test kept;
  - the LMCache tier is PER BLOCK: GLM53_LMCACHE_DISK_DIR, the fs remote tier (GLM53_LMCACHE_REMOTE_URL), GLM53_KDA_CKPT_DIR and
    GLM53_KDA_CKPT_LOCAL_DIR under a blk<N> directory of this kit stamp (no store or checkpoint is shared between trays);
  - NO image in the env job's launch block: the image root + pin are PER BLOCK and given at launch by glx_dp4_up.sh (GLX_IMAGE_ROOT +
    GLX_PIN from dp4_images.conf). blk0 / blk1's harvest moves the bank-adjacent L1 column, so the 1530 image (the host-8chip / blk2 /
    blk3 map) fails there at upload #179: blk0 / blk1 take the 629a map or a device build of their own (dp4/README.md);
    an env-job image_root would win over GLX_IMAGE_ROOT in glx_pd_role.sh, so none is written;
  - the in-flight policy, max_model_len 1,048,576 and the rest of the served env come from the form's env job unchanged (P1 / D24 =
    its GLM53_TT_CFG_EXTRA decode_interleave_prefill_steps 1 / decode_interleave_decode_steps 24; asserted);
  - refused: any speculative / DFlash2 key on, WS_O != 0, the R1b / B11 knobs, an export key set, another plugin or tree.
usage: mk_dp4_envjobs.py <out root> [--form t6d|t6d2|f2|t6di|f2i|f2i3|f2i3b|f2i3c|f2i3cxb|f2i3d|f2i3d2|f2i3d3|f2i3d4|f2i3d5|f2i3d6|f2i3d7|smu2g|smu2v4m2xb|smu2v4m2xbgc|smu2v4m2xbnp|smu2v4m2xbv4|smu2v4m2xbv5|smu2v4m2xbv6|smu2v4m2xbv7|df2gv4c4xb|smu2g4|smu2g4n4|df2g|df2g4|df2g4v2] [--base-envjob P --base-target P] [--sha <tree sha11>] [--plugin <src>:<sha7>] [--stamp <UTC>] [--pin <image pin>]
  -> <out root>/glm53/work-s9/envjobs/... and <out root>/glm53-flash-bringup/doc/ttnn/09-vllm/scripts/targets/...; one line per file."""
import json
import os
import re
import subprocess
import sys
import time

REPL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = "/opt/glm53/kit"
DATA = "/opt/glm53/data"
ENVJOBS = "/opt/glm53/work-s9/envjobs"
PLUGIN, PLUGIN_SHA = "/opt/glm53/srv/s9hp5bc/src", "5bcf424"
LMC, LMC_SHA = "/opt/glm53/LMCache-tt-s9k4d", "0e45f0ec"
FORMS = {"t6d": ("n29h7p2hk4d8t6d", "/opt/glm53/tt-metal-s9cfk4d8t6d", "5aec76ac6c7",
                 {"GLM53_PMK_DSA_POST": "1", "GLM53_PMK_RS8_PAIRFWD": "p15", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1"}),
         "f2": ("n29h7p2hk4d8t5wp6f2", "/opt/glm53/tt-metal-s9cfk4d8t5w", "dbfa7ac6290",
                {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1"}),
         # program 2026-10-28T13:09:14Z (operator: "I NEED SPEC DECODE"): smu2g = the t6d env + the joint tree (5aec76ac6c7 + df2tap +
         # 17's shmu twin commits) + 17's plugin (policy ON, MTP K=3, DFlash2 K=7 seeded); its sha / plugin come as --sha / --plugin
         # program 2026-10-28T13:41:36Z (1): t6di = the t6d form + 12's plugin 878878b (srv/s12hitpath, --plugin) + the shortest-first
         # prefill interleave GLM53_PREFILL_SHORT_FIRST=32768 (MAX 4 / RETRY 8 defaults)
         "t6di": ("n29h7p2hk4d8t6di", "/opt/glm53/tt-metal-s9cfk4d8t6d", "5aec76ac6c7",
                  {"GLM53_PMK_DSA_POST": "1", "GLM53_PMK_RS8_PAIRFWD": "p15", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1",
                   "GLM53_PREFILL_SHORT_FIRST": "32768"}),
         # program [session 10] 2026-10-28T14:36:32Z RULING (dsapost RED): f2i = the f2 form + 12's plugin 878878b (--plugin) + the
         # shortest-first prefill interleave GLM53_PREFILL_SHORT_FIRST=32768: the fleet form (it replaces t6di)
         "f2i": ("n29h7p2hk4d8t5wp6f2i", "/opt/glm53/tt-metal-s9cfk4d8t5w", "dbfa7ac6290",
                 {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768"}),
         # program [session 10] 2026-10-28T16:16:13Z / 16:17:11Z: f2i3 = the f2i form + 12's plugin v3 (--plugin; 878878b + the
         # deferred-request admission fix) with GLM53_KV_DEVICE_THREAD_AUDIT UNSET (= the v2 env): the final fleet form
         "f2i3": ("n29h7p2hk4d8t5wp6f2i3", "/opt/glm53/tt-metal-s9cfk4d8t5w", "dbfa7ac6290",
                  {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768",
                   "GLM53_KV_DEVICE_THREAD_AUDIT": None}),
         # program 2026-10-28T13:55:25Z: df2g = 17's DFlash2 form (13:49:55Z) on 16's 5737ddd3e48 (the DFlash2 multi-user layouts),
         # unseeded K=7 with the row policy, the same 131,072 cap as smu2g; sha / plugin via --sha / --plugin
         "df2g": ("n29h7p2hdf2g", "/opt/glm53/tt-metal-s9cfdf2g", None,
                  {"GLM53_PMK_DSA_POST": "1", "GLM53_PMK_RS8_PAIRFWD": "p15", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1",
                   "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "dflash2", "GLM53_SPEC_ROW_POLICY": "1"}),
         "smu2g": ("n29h7p2hsmu2g", "/opt/glm53/tt-metal-s9cfsmu2g", None,
                   {"GLM53_PMK_DSA_POST": "1", "GLM53_PMK_RS8_PAIRFWD": "p15", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1",
                    "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "mtp", "GLM53_SPEC_ROW_POLICY": "1"}),
         # program [program] 2026-10-28T18:21:36Z / 18:24:20Z: f2i3b = f2i3 EXACTLY + 12's plugin v3b2 efb8c79 (the overtake on the priority heap)
         "f2i3b": ("n29h7p2hk4d8t5wp6f2i3b", "/opt/glm53/tt-metal-s9cfk4d8t5w", "dbfa7ac6290",
                   {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768",
                    "GLM53_KV_DEVICE_THREAD_AUDIT": None}),
         # program [program] 2026-10-28T19:22:27Z / 19:22:57Z + [session 10] 19:23:06Z: f2i3c = f2i3b + 12's plugin v3b3 534ea57 (the
         # SHORT_FIRST pacing, INFLIGHT default 2) + 7a's CB guard tree 4f2aa1d790d (= dbfa7ac6290 + the guard) + 12's capped cold tier
         # (LMCache 943926cb; REMOTE_GB 200 / FREE_FLOOR 300 / GRACE 120 per block): the overnight delivery form
         "f2i3c": ("n29h7p2hk4d8t5wp6f2i3c", "/opt/glm53/tt-metal-s9cfk4d8t5wg", "4f2aa1d790d",
                   {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768",
                    "GLM53_KV_DEVICE_THREAD_AUDIT": None, "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300",
                    "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # program [session 10] 2026-10-29T00:24:57Z (the width-8 stall fix composed into every served form): f2i3cxb = f2i3c + 12's plugin
         # v3b4 c814cdb (the SHORT_FIRST pacing at the device; recorded 23:41:25Z) on 8a9c2d5b5fe = 4f2aa1d790d + 14b's cq_prefetch.cpp drain
         # 3fc9517c3d0 (one JIT dispatch kernel file, md5 40e8ebb3; glx_k4d7lg_tree_prep.sh farms tt_metal/ with it): the morning fleet form
         "f2i3cxb": ("n29h7p2hk4d8t5wp6f2i3cxb", "/opt/glm53/tt-metal-s9cfk4d8t5wgxb", "8a9c2d5b5fe",
                     {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768",
                      "GLM53_KV_DEVICE_THREAD_AUDIT": None, "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300",
                      "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # [program] 2026-10-29T00:55:25Z: f2i3d = f2i3cxb EXACTLY (8a9c2d5b5fe, the capped tier, env keys unchanged) with 12's plugin v3b5
         # 2c8258b (= v3b4 c814cdb + gc_tuning: the engine GC policy) as srv/s12v3b5
         "f2i3d": ("n29h7p2hk4d8t5wp6f2i3d", "/opt/glm53/tt-metal-s9cfk4d8t5wgxb", "8a9c2d5b5fe",
                   {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768",
                    "GLM53_KV_DEVICE_THREAD_AUDIT": None, "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300",
                    "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # [program] 2026-10-29T01:43:37Z: f2i3d2 = dp4_f2i3d_v2 = f2i3d EXACTLY with 12's v3b6 dee1f07 (v3b5 + the admit gauges on drain / abort) as
         # srv/s12v3b6; a new tag so v1's H-mock verdicts cannot vouch for it
         "f2i3d2": ("n29h7p2hk4d8t5wp6f2i3d2", "/opt/glm53/tt-metal-s9cfk4d8t5wgxb", "8a9c2d5b5fe",
                   {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768",
                    "GLM53_KV_DEVICE_THREAD_AUDIT": None, "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300",
                    "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # program [program] 2026-10-28T17:54:38Z: df2g4 = the f2i3 fleet form (f2 line, no dsapost) + 17's 9e8502dfa53 + plugin c9d499e +
         # the DFlash2 K=7 env of 17 16:46:37Z (SLOT_PAGES unset, CACHE_CAP 32768, MULTI_CAP 16384), the 131,072 cap
         "df2g4": ("n29h7p2hdf2g4", "/opt/glm53/tt-metal-s9cfk4d8t5wdf2gv4", None,
                   {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                    "GLM53_PMK_DSA_POST": None, "GLM53_DF2_SLOT_PAGES": None, "GLM53_DF2_CACHE_CAP": "32768", "GLM53_DF2_MULTI_CAP": "16384",
                    "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "dflash2", "GLM53_SPEC_ROW_POLICY": "1"}),
         # program [session 10] 2026-10-28T20:21:49Z: t6d2 = the f2i3b fleet form + dsapost2 on 16's fb8e99f9068 (f2 + dsapost + the dsapost2
         # line + 7a's guard; bitwise on the op path at TP8) + DSA_POST=1 + RS8_OPSUM=dsa + the capped tier, plugin efb8c79
         "t6d2": ("n29h7p2hk4d8t6d2", "/opt/glm53/tt-metal-s9cfk4d8t6d2", "fb8e99f9068",
                  {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                   "GLM53_PMK_DSA_POST": "1", "GLM53_PMK_RS8_OPSUM": "dsa", "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300",
                   "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # program [session 10] 2026-10-28T19:25:41Z (1) / (2) + [program] 19:29:09Z: df2g4v2 = df2g4 + MULTI_WS=16 (the K=7 W32 L1 floor; at W16
         # the 4- and 8-user cells decode plain) on 17's 9e8502dfa53 + 7a's guard (fd34cb257c0); smu2g4n4 = the mesh m4 = smu2v4 (the f2i3 fleet
         # form + 17's MTP K=3 env) on 17's ef5ed397611 (GLM53_SPEC_MULTI_MAX_USERS) + 7a's guard (75cc2195658) at N=4; both + the capped tier
         "df2g4v2": ("n29h7p2hdf2g4v2", "/opt/glm53/tt-metal-s9cfk4d8t5wdf2gv4g", "fd34cb257c0",
                     {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                      "GLM53_PMK_DSA_POST": None, "GLM53_DF2_SLOT_PAGES": None, "GLM53_DF2_CACHE_CAP": "32768", "GLM53_DF2_MULTI_CAP": "16384",
                      "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "dflash2", "GLM53_SPEC_ROW_POLICY": "1", "GLM53_SPEC_MULTI_WS": "16", "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300", "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         "smu2g4n4": ("n29h7p2hsmu2g4n4", "/opt/glm53/tt-metal-s9cfk4d8t5wsmu2v4m4", "75cc2195658",
                      {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                       "GLM53_PMK_DSA_POST": None, "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "mtp", "GLM53_SPEC_ROW_POLICY": "1",
                       "GLM53_SPEC_MULTI_MAX_USERS": "4", "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300", "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # [program] 2026-10-29T00:49:56Z (the m2 S4 PASS 00:49:28Z: users 8 95.1 pct / 16 97.6 pct, lone x1.41, 2u x1.60): smu2v4m2xb = the
         # host-8chip m2 target n29h7p2hk4d8t5wsmu2v4m2 EXACTLY (= smu2g4n4 with GLM53_SPEC_MULTI_MAX_USERS=2) on 9380d04d8a2 = 75cc2195658 + 14b's
         # cq_prefetch.cpp drain 3fc9517c3d0 (md5 40e8ebb3) + the capped tier; the blk2 spec tray of the morning roll
         "smu2v4m2xb": ("n29h7p2hk4d8t5wsmu2v4m2xb", "/opt/glm53/tt-metal-s9cfk4d8t5wsmu2v4m4xb", "9380d04d8a2",
                        {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                         "GLM53_PMK_DSA_POST": None, "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "mtp", "GLM53_SPEC_ROW_POLICY": "1",
                         "GLM53_SPEC_MULTI_MAX_USERS": "2", "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300", "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # [program] 2026-10-29T01:04:56Z: smu2v4m2xbgc = smu2v4m2xb EXACTLY with 12's plugin 3272a90 (= c9d499e + the v3b5 engine GC policy) as
         # srv/s9spec6; a new tag so the v1 blk2 H-mock verdict cannot vouch for it (tar dp4_smu2v4m2xb_v2)
         "smu2v4m2xbgc": ("n29h7p2hk4d8t5wsmu2v4m2xbgc", "/opt/glm53/tt-metal-s9cfk4d8t5wsmu2v4m4xb", "9380d04d8a2",
                        {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                         "GLM53_PMK_DSA_POST": None, "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "mtp", "GLM53_SPEC_ROW_POLICY": "1",
                         "GLM53_SPEC_MULTI_MAX_USERS": "2", "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300", "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # program [session 10] 2026-10-29T01:30:57Z (the c4 row PASSES the S4 bar: users 8 97.2 pct / 16 96.2 pct, lone x1.95, greedy-equal):
         # df2gv4c4xb = the host-8chip df2gv4c4 target EXACTLY (DFlash2 K=7, MULTI_WS 16, plugin c4c1358) on 206ac7d0f4a = fd34cb257c0 + the cq_prefetch drain
         "df2gv4c4xb": ("n29h7p2hk4d8t5wdf2gv4c4xb", "/opt/glm53/tt-metal-s9cfk4d8t5wdf2gv4gxb", "206ac7d0f4a",
                        {"GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                         "GLM53_PMK_DSA_POST": None, "GLM53_DF2_SLOT_PAGES": None, "GLM53_DF2_CACHE_CAP": "32768", "GLM53_DF2_MULTI_CAP": "16384",
                         "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "dflash2", "GLM53_SPEC_ROW_POLICY": "1", "GLM53_SPEC_MULTI_WS": "16",
                         "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300", "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # [program] 2026-10-29T02:49:50Z OPERATOR GO (1): f2i3d3 / smu2v4m2xbnp = f2i3d2 / smu2v4m2xbgc EXACTLY + GLM53_ADMIT_KV_BUDGET=1
         # GLM53_ADMIT_RESERVE_TOKENS=0 (the no-preempt reserve) -- tars dp4_f2i3d_v3 / dp4_smu2v4m2xb_v3
         "f2i3d3": ("n29h7p2hk4d8t5wp6f2i3d3", "/opt/glm53/tt-metal-s9cfk4d8t5wgxb", "8a9c2d5b5fe",
                   {"GLM53_ADMIT_KV_BUDGET": "1", "GLM53_ADMIT_RESERVE_TOKENS": "0", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768",
                    "GLM53_KV_DEVICE_THREAD_AUDIT": None, "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300",
                    "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         "smu2v4m2xbnp": ("n29h7p2hk4d8t5wsmu2v4m2xbnp", "/opt/glm53/tt-metal-s9cfk4d8t5wsmu2v4m4xb", "9380d04d8a2",
                        {"GLM53_ADMIT_KV_BUDGET": "1", "GLM53_ADMIT_RESERVE_TOKENS": "0", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                         "GLM53_PMK_DSA_POST": None, "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "mtp", "GLM53_SPEC_ROW_POLICY": "1",
                         "GLM53_SPEC_MULTI_MAX_USERS": "2", "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300", "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # [program] 2026-10-29T02:49:50Z OPERATOR GO (2) + 02:57:03Z (3): the v4 forms = the v3 forms (df2gv4c4xb + the reserve keys) with the
         # fixed plugins: PLAIN adbc99f (12's v3b8 = aafe5d8 (B) + the (A) hand port) as s12v3b8, SPEC 0880fe1 (= ac0386e (B) + 17's 1139367 (A)) as s9spec7
         "f2i3d4": ("n29h7p2hk4d8t5wp6f2i3d4", "/opt/glm53/tt-metal-s9cfk4d8t5wgxb", "8a9c2d5b5fe",
                   {"GLM53_ADMIT_KV_BUDGET": "1", "GLM53_ADMIT_RESERVE_TOKENS": "0", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768",
                    "GLM53_KV_DEVICE_THREAD_AUDIT": None, "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300",
                    "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         "smu2v4m2xbv4": ("n29h7p2hk4d8t5wsmu2v4m2xbv4", "/opt/glm53/tt-metal-s9cfk4d8t5wsmu2v4m4xb", "9380d04d8a2",
                        {"GLM53_ADMIT_KV_BUDGET": "1", "GLM53_ADMIT_RESERVE_TOKENS": "0", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                         "GLM53_PMK_DSA_POST": None, "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "mtp", "GLM53_SPEC_ROW_POLICY": "1",
                         "GLM53_SPEC_MULTI_MAX_USERS": "2", "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300", "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         "df2gv4c4xbv4": ("n29h7p2hk4d8t5wdf2gv4c4xbv4", "/opt/glm53/tt-metal-s9cfk4d8t5wdf2gv4gxb", "206ac7d0f4a",
                        {"GLM53_ADMIT_KV_BUDGET": "1", "GLM53_ADMIT_RESERVE_TOKENS": "0", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                         "GLM53_PMK_DSA_POST": None, "GLM53_DF2_SLOT_PAGES": None, "GLM53_DF2_CACHE_CAP": "32768", "GLM53_DF2_MULTI_CAP": "16384",
                         "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "dflash2", "GLM53_SPEC_ROW_POLICY": "1", "GLM53_SPEC_MULTI_WS": "16",
                         "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300", "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # [program] 2026-10-29T03:07:42Z / 03:08:37Z: v5 = v4 EXACTLY with the tree on 17's (A) tree sha (generator PageRowShort): NOT in the roll, block restarts after it
         "f2i3d5": ("n29h7p2hk4d8t5wp6f2i3d5", "/opt/glm53/tt-metal-s9cfk4d8t5wgxbrs", "95e7ef9a315",
                   {"GLM53_ADMIT_KV_BUDGET": "1", "GLM53_ADMIT_RESERVE_TOKENS": "0", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768",
                    "GLM53_KV_DEVICE_THREAD_AUDIT": None, "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300",
                    "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         "smu2v4m2xbv5": ("n29h7p2hk4d8t5wsmu2v4m2xbv5", "/opt/glm53/tt-metal-s9cfk4d8t5wsmu2v4m4xbrs", "caea4b40a1d",
                        {"GLM53_ADMIT_KV_BUDGET": "1", "GLM53_ADMIT_RESERVE_TOKENS": "0", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                         "GLM53_PMK_DSA_POST": None, "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "mtp", "GLM53_SPEC_ROW_POLICY": "1",
                         "GLM53_SPEC_MULTI_MAX_USERS": "2", "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300", "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # [program] 2026-10-29T03:17:25Z: v6 = v4 EXACTLY with the HARDENED plugins (PLAIN e330b8a as s12v3b8h, SPEC 9ca7caf as s9spec7h): the roll's forms
         "f2i3d6": ("n29h7p2hk4d8t5wp6f2i3d6", "/opt/glm53/tt-metal-s9cfk4d8t5wgxb", "8a9c2d5b5fe",
                   {"GLM53_ADMIT_KV_BUDGET": "1", "GLM53_ADMIT_RESERVE_TOKENS": "0", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768",
                    "GLM53_KV_DEVICE_THREAD_AUDIT": None, "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300",
                    "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         "smu2v4m2xbv6": ("n29h7p2hk4d8t5wsmu2v4m2xbv6", "/opt/glm53/tt-metal-s9cfk4d8t5wsmu2v4m4xb", "9380d04d8a2",
                        {"GLM53_ADMIT_KV_BUDGET": "1", "GLM53_ADMIT_RESERVE_TOKENS": "0", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                         "GLM53_PMK_DSA_POST": None, "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "mtp", "GLM53_SPEC_ROW_POLICY": "1",
                         "GLM53_SPEC_MULTI_MAX_USERS": "2", "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300", "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # [program] 2026-10-29T03:27:12Z post-roll fold: v7 = v5 (17's generator PageRowShort trees) with 12's v7 plugins (PLAIN de99edb as s12v3b9, SPEC ea731dd as s9spec8)
         "f2i3d7": ("n29h7p2hk4d8t5wp6f2i3d7", "/opt/glm53/tt-metal-s9cfk4d8t5wgxbrs", "95e7ef9a315",
                   {"GLM53_ADMIT_KV_BUDGET": "1", "GLM53_ADMIT_RESERVE_TOKENS": "0", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768",
                    "GLM53_KV_DEVICE_THREAD_AUDIT": None, "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300",
                    "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         "smu2v4m2xbv7": ("n29h7p2hk4d8t5wsmu2v4m2xbv7", "/opt/glm53/tt-metal-s9cfk4d8t5wsmu2v4m4xbrs", "caea4b40a1d",
                        {"GLM53_ADMIT_KV_BUDGET": "1", "GLM53_ADMIT_RESERVE_TOKENS": "0", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1", "GLM53_PREFILL_SHORT_FIRST": "32768", "GLM53_KV_DEVICE_THREAD_AUDIT": None,
                         "GLM53_PMK_DSA_POST": None, "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "mtp", "GLM53_SPEC_ROW_POLICY": "1",
                         "GLM53_SPEC_MULTI_MAX_USERS": "2", "GLM53_LMCACHE_REMOTE_GB": "200", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB": "300", "GLM53_LMCACHE_REMOTE_GRACE_S": "120"}),
         # program [program] 2026-10-28T16:35:18Z: smu2g4 = 17's v4 9df7febab7c (Python only on c5104136582) with the smu2g v2 env
         "smu2g4": ("n29h7p2hsmu2g4", "/opt/glm53/tt-metal-s9cfsmu2g4", None,
                    {"GLM53_PMK_DSA_POST": "1", "GLM53_PMK_RS8_PAIRFWD": "p15", "GLM53_PMK_RTAIL_BATCH": "1", "PMK_P12_FAST_SCAN": "1",
                     "GLM53_MAX_MODEL_LEN": "131072", "GLM53_SPEC_KIND": "mtp", "GLM53_SPEC_ROW_POLICY": "1"})}  # 17 13:49:55Z: the MTP form at 131,072
SPEC_FORMS = {"smu2g", "smu2g4", "df2g", "df2g4", "df2g4v2", "smu2g4n4", "smu2v4m2xb", "smu2v4m2xbgc", "smu2v4m2xbnp", "smu2v4m2xbv4", "smu2v4m2xbv5", "smu2v4m2xbv6", "smu2v4m2xbv7", "df2gv4c4xb", "df2gv4c4xbv4"}
# 19:22:27Z: a form's own LMCache tree (default LMC above): f2i3c runs 12's capped cold tier
FORM_LMC = {f: ("/opt/glm53/LMCache-tt-s9cap", "943926cb") for f in ("f2i3c", "f2i3cxb", "f2i3d", "f2i3d2", "f2i3d3", "smu2v4m2xbnp", "f2i3d4", "smu2v4m2xbv4", "f2i3d5", "smu2v4m2xbv5", "f2i3d6", "smu2v4m2xbv6", "f2i3d7", "smu2v4m2xbv7", "df2gv4c4xbv4", "df2g4v2", "smu2g4n4", "smu2v4m2xb", "smu2v4m2xbgc", "df2gv4c4xb", "t6d2")}  # forms whose speculative keys are the point (MTP / DFlash2 ON); every other form refuses them


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


out = sys.argv[1]
form = arg("--form", "t6d")
TAG, TREE, TREE_SHA, FORM_KEYS = FORMS[form]
TREE_SHA = arg("--sha", TREE_SHA)
LMC, LMC_SHA = FORM_LMC.get(form, (LMC, LMC_SHA))
if not TREE_SHA:
    sys.exit(f"form {form}: give the tree sha (--sha)")
if arg("--plugin"):  # <src>:<sha7>, e.g. /opt/glm53/srv/s9smu2g/src:abc1234 (the spec form's own plugin)
    PLUGIN, PLUGIN_SHA = arg("--plugin").rsplit(":", 1)
# --base-envjob / --base-target (18:42Z): a composed base outside the host-8chip target path (the host-8chip mock owns that path)
base = arg("--base-envjob", f"{ENVJOBS}/int27{TAG}_img_env.done")
stamp = arg("--stamp") or time.strftime("%Y%m%dT%H%M%S", time.gmtime())
pin = arg("--pin", "fde4697e51b2-tp-ds-v1-2x4-1530493535ed")
ejd = os.path.join(out, "glm53/work-s9/envjobs")
td = os.path.join(out, "glm53-flash-bringup/doc/ttnn/09-vllm/scripts/targets")
os.makedirs(ejd, exist_ok=True)
os.makedirs(td, exist_ok=True)
head = subprocess.run(["git", "-C", TREE, "rev-parse", "--short=11", "HEAD"], capture_output=True, text=True).stdout.strip()
if head != TREE_SHA:
    sys.exit(f"{TREE} is at {head or '?'}, not {TREE_SHA}")
gbase = os.path.join(ejd, f"int27{TAG}_img_env_glx.done")
subprocess.run([sys.executable, "-I", os.path.join(REPL, "mk_glx_envjob.py"), base, gbase, "--pin", pin], check=True)
gj = json.load(open(gbase))
genv = gj["args"]["env"]
# the served form of record, asserted on the base before anything is derived from it
bad = []
want = {"GLM53_PMK_WS_O": "0", "GLM53_KDA_CKPT_COLLECT_SERIAL": "0", "GLM53_KV_DEVICE_THREAD_AUDIT": "1", "GLM53_MAX_MODEL_LEN": "1048576",
        "GLM53_PF_TAIL_SPLIT_SKIP": "1", "GLM53_PMK_PFS_ROUTE": "1", "GLM53_PMK_PFS_MC_PKT": "4", **FORM_KEYS}
bad += [f"{k}={genv.get(k)!r} (want {v})" for k, v in want.items() if genv.get(k) != v]
for k in ("GLM53_KDA_CKPT_RESOLVERS", "GLM53_KV_LOOKUP_DEFER_KDA", "GLM53_KV_STORE_YIELD_TO_STATE_SAVE", "GLM53_KV_STORE_YIELD_MAX_S",
          "GLM53_KDA_CKPT_RESOLVER_SAMPLE_S", "GLM53_PMK_AGG_UNICAST", "GLM53_FS_PKT_X", "GLM53_FS_PKT_SCOPE"):
    if genv.get(k) not in (None, "", "0"):
        bad.append(f"{k} set (withdrawn: program 2026-10-28T11:09:03Z / 11:21:07Z)")
ks = genv.get("GLM53_STACK_KNOB_SET", "")
if "GLM53_RTAIL_SCAN=split" in ks or "GLM53_REC_B11=" in ks:
    bad.append("R1b / B11 in the knob set")
spec_on = sorted(k for k, v in genv.items() if (k.startswith(("GLM53_DF2", "GLM53_MTP", "GLM53_DFLASH")) or k == "GLM53_SPEC_ENABLE")
                 and str(v) not in ("", "0", "off"))
if spec_on and form not in SPEC_FORMS:
    bad.append(f"speculative / DFlash2 keys on: {spec_on} (MTP off, DFlash2 off)")
cfg = genv.get("GLM53_TT_CFG_EXTRA", "")
if '"decode_interleave_prefill_steps": 1' not in cfg or '"decode_interleave_decode_steps": 24' not in cfg:
    bad.append(f"the in-flight policy is not P1 / D24: GLM53_TT_CFG_EXTRA={cfg!r}")
if os.path.basename(genv.get("GLM53_TP_IMAGE_DIR", "")) != pin:
    bad.append(f"GLM53_TP_IMAGE_DIR {genv.get('GLM53_TP_IMAGE_DIR')} is not the pin {pin}")
# the plugin and the tree are the base TARGET's exports (the env job does not carry them)
src_t = open(arg("--base-target", f"{WS}/doc/ttnn/09-vllm/scripts/targets/int27{TAG}_resident.sh")).read()
for line in (f"export GLM53_PLUGIN_SRC={PLUGIN}", f"export GLM53_S9_TREE={TREE}"):
    if line not in src_t.split("\n"):
        bad.append(f"the target int27{TAG}_resident.sh lacks '{line}'")
if form not in SPEC_FORMS and re.search(r"^export GLM53_SPEC_CONFIG=", src_t, re.M):
    bad.append(f"the target int27{TAG}_resident.sh sets a speculative config")
if form in SPEC_FORMS and not re.search(r"^export GLM53_SPEC_CONFIG=", src_t, re.M):
    bad.append(f"the spec form's target int27{TAG}_resident.sh has no GLM53_SPEC_CONFIG")
if bad:
    sys.exit(f"{base}: not the served form of record: " + "; ".join(bad))
for N in range(4):
    btag = f"{TAG}dp4b{N}"
    dirs = {"GLM53_KDA_CKPT_DIR": f"{DATA}/kda_ckpt/dp4/{TAG}-blk{N}-{stamp}",
            "GLM53_LMCACHE_REMOTE_URL": f"fs://localhost:0{DATA}/lmcache_cold/dp4/{TAG}-blk{N}-{stamp}",
            "GLM53_LMCACHE_DISK_DIR": f"/opt/glm53/lmcache_nvme_dp4/{TAG}-blk{N}-{stamp}",
            "GLM53_KDA_CKPT_LOCAL_DIR": f"/opt/glm53/kda_ckpt_local_dp4/{TAG}-blk{N}-{stamp}"}
    for k in dirs:
        assert k in genv, f"the base env job has no {k}"
    keys = {**dirs, "GLM53_LMCACHE_TREE": LMC, "GLM53_PLUGIN_SRC": PLUGIN}
    env = {**genv, **keys}
    ej = f"{ENVJOBS}/int27{btag}pdH_img_env.done"
    job = {"script": gj.get("script", ""), "args": {"env": env},
           "stage9_overlay": {"base": f"{ENVJOBS}/int27{TAG}_img_env_glx.done", "keys": keys, "dropped": [],
                              "authority": f"program 2026-10-28T12:20:55Z / 12:25:35Z (operator delivery: TP8 x DP4 on the Galaxy, form {form} = {TAG}); "
                                           f"tray blk{N} of DP4 stamp {stamp}: the block's own LMCache tier + KDA checkpoint dirs + image root",
                              "role": "H", "block": N, "dp4_stamp": stamp, "dp4_form": form,
                              "launch": {"plugin_src": PLUGIN, "plugin_sha": PLUGIN_SHA, "lmcache_tree": LMC, "lmcache_sha": LMC_SHA,
                                         "tree": TREE}}}
    p = os.path.join(ejd, f"int27{btag}pdH_img_env.done")
    json.dump(job, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)
    # the tray's target: the form's served target with the tray identity, its env job, the per-block dirs; strict self-test kept
    t = src_t
    for a_, b_ in ((f"GLM53_TARGET=int27{TAG} GLM53_RUN=resident_int27{TAG}_", f"GLM53_TARGET=int27{btag} GLM53_RUN=resident_int27{btag}_"),
                   (f"GLM53_14B_ENV_FILE={ENVJOBS}/int27{TAG}_img_env.done", f"GLM53_14B_ENV_FILE={ej}")):
        assert t.count(a_) == 1, f"{TAG} target: {a_!r} x{t.count(a_)}"
        t = t.replace(a_, b_, 1)
    t, n = re.subn(r"^export GLM53_SELFTEST=strict GLM53_SELFTEST_OUT=\S+$",
                   f"export GLM53_SELFTEST=strict GLM53_SELFTEST_OUT=/opt/glm53/work-s9/runs/glxdp4_{btag}_selftest_$(date -u +%Y%m%dT%H%M%S).json",
                   t, count=1, flags=re.M)
    assert n == 1, "no strict GLM53_SELFTEST line"
    for k, v in dirs.items():
        t, n = re.subn(rf"^export {k}=\S*$", lambda m_: f"export {k}={v}", t, flags=re.M)
        assert n == 1, f"{k}: {n} export lines in the target"
    t, n = re.subn(r"^(export GLM53_CAPTURE_GATE='PASS[^']*)'(.*)$", rf"\1 | the Galaxy DP4 tray blk{N} (form {form}, stamp {stamp})'\2", t, count=1, flags=re.M)
    assert n == 1, "no single PASS capture gate line"
    hdr = (f"#!/usr/bin/env bash\n# The Galaxy DP4 tray blk{N} (program 2026-10-28T12:20:55Z / 12:25:35Z): generated by replicate/dp4/mk_dp4_envjobs.py\n"
           f"# --form {form} from int27{TAG}_resident.sh (stamp {stamp}); launched by pd/glx_pd_role.sh (GLX_PD_ROLE=H GLX_PD_TAG={btag}) through\n"
           f"# replicate/dp4/glx_dp4_up.sh; its mock: glx_image_mock.sh GLX_TAG={btag} GLX_PD_ROLE=H on the blk{N} image root + descriptor.\n")
    t = hdr + t.split("\n", 1)[1]
    tp = os.path.join(td, f"int27{btag}_resident.sh")
    open(tp, "w").write(t)
    os.chmod(tp, 0o755)
    print(f"blk{N}: {p} ({len(keys)} overlay keys); {tp}")
print(f"form {form} = {TAG} (tree {TREE} {TREE_SHA}); base (Galaxy form): {gbase}; pin {pin}; stamp {stamp}")
