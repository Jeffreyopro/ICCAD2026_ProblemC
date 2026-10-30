# ICCAD 2026 FloorSet — 三階段 Fast-SA 實驗結果

本檔案由 `work/make_notes.py` 自動產生，所有數字來自 `work/bench_*.json`。

## 1. 重現指令

```powershell
cd C:\Floorset\FloorSet\iccad2026contest
# (Windows) 編譯 C 擴充；需要 MSYS2 UCRT64 gcc
powershell -ExecutionPolicy Bypass -File .\build_bstar_windows.ps1
# 官方評分（單一案例快速驗證）
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 95 --quick
# 20 案例分層抽樣 benchmark（單一 restart、固定 seed）
python C:\Floorset\work\benchmark.py modified_reduced original_equal modified_long
# C 擴充 A/B 對照（關閉 C 路徑）
$env:BENCH_CASES="0,40,70,95"; $env:FLOORSET_NO_CYTHON="1"; python C:\Floorset\work\benchmark.py mod_py
```

## 2. 整體統計（20 案例）

| 設定 | 平均成本 | 品質因子 | HPWL 差距 | 面積差距 | 軟約束 | 可行解 | 平均時間 (s) | 總移動 | Stage I / II / III |
|---|---|---|---|---|---|---|---|---|---|
| 三階段（縮減） | 2.1866 | 2.1866 | 1.0180 | 0.5284 | 0.10028 | 20/20 | 3.92 | 12059 | 1167 / 8852 / 2040 |
| 原始 Fast-SA | 2.7132 | 2.7132 | 1.4194 | 0.9522 | 0.10834 | 20/20 | 3.52 | 11295 | 1167 / 8712 / 1416 |
| 三階段（長） | 2.1536 | 2.1536 | 1.0058 | 0.5090 | 0.09868 | 20/20 | 4.08 | 12754 | 1167 / 8852 / 2735 |

- 三階段 vs 原始：平均成本 19.4% 改善，HPWL 差距降低 28.3%，面積差距降低 44.5%，逐案例勝出 16/20。
- Stage III 縮減：移動數 12754 → 12059（-5.4%），Stage III 步數 2735 → 2040（-25.4%），平均成本 2.1536 → 2.1866。

## 3. 逐案例成本

| 案例 | 區塊數 | 三階段 | 原始 | 差異 |
|---|---|---|---|---|
| 0 | 21 | 2.0526 | 2.0087 | +2.2% |
| 5 | 26 | 2.0529 | 2.1489 | -4.5% |
| 10 | 31 | 1.8717 | 2.5126 | -25.5% |
| 15 | 36 | 1.8287 | 1.9216 | -4.8% |
| 20 | 41 | 1.6554 | 2.6805 | -38.2% |
| 25 | 46 | 3.4208 | 3.0738 | +11.3% |
| 30 | 51 | 2.4412 | 3.1096 | -21.5% |
| 35 | 56 | 1.7438 | 2.5847 | -32.5% |
| 40 | 61 | 1.8542 | 2.5134 | -26.2% |
| 45 | 66 | 1.8754 | 1.6991 | +10.4% |
| 50 | 71 | 1.8430 | 2.5866 | -28.7% |
| 55 | 76 | 2.2338 | 3.6436 | -38.7% |
| 60 | 81 | 2.0046 | 2.7857 | -28.0% |
| 65 | 86 | 2.4159 | 3.0887 | -21.8% |
| 70 | 91 | 3.1882 | 2.6073 | +22.3% |
| 75 | 96 | 2.2369 | 3.0853 | -27.5% |
| 80 | 101 | 1.8052 | 3.1802 | -43.2% |
| 85 | 106 | 2.3941 | 3.1032 | -22.9% |
| 90 | 111 | 2.4560 | 2.8850 | -14.9% |
| 95 | 116 | 2.3580 | 3.0463 | -22.6% |

## 5. 平行 worker 驗證

```text
[Cython] bstar_core active
[solve] Running 4 restarts in parallel on 12 cores
  Phase 3 [1/4] restarts complete
  ...
  Winner: restart 1 (cost=1.9180)
Total Score: 1.6810
```

以 `FLOORSET_RESTARTS=4` 執行官方評分流程，4 個子行程皆完成並回傳，父行程正確選出最佳 restart。

## 6. 設定對照

| 參數 | 原始提交 | 本專題預設 |
|---|---|---|
| FLOORSET_MAX_LEVELS | 48 | 26 |
| FLOORSET_STAGE3_SCALE | 1.5 | 1.0 |
| FLOORSET_PHASE3_FACTOR | 0.25 | 0.15 |
| FLOORSET_PHASE3_PATIENCE | 3 | 2 |
| FLOORSET_SCHEDULE | original | modified |

其他可用環境變數：`FLOORSET_RESTARTS`、`FLOORSET_PARALLEL`、`FLOORSET_STAGE2_SCALE`、`FLOORSET_FASTSA_K`、`FLOORSET_SEED`、`FLOORSET_STATS`、`FLOORSET_NO_CYTHON`。
