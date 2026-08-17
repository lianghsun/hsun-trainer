# hsun-trainer

在 Claude Code 裡完成 LLM 訓練全流程的 skill plugin：
**繼續預訓練 (CPT) → 監督式微調 (SFT) → GRPO/RLVR → 評測**，
以繁體中文 (zh-TW) 與台灣資料集為第一優先。

以 [TRL](https://github.com/huggingface/trl) 為訓練後端，
以 [ai-twinkle/Eval](https://github.com/ai-twinkle/Eval) 為評測後端，
同一份 recipe 可跑在自有 GPU 機器或 Hugging Face Jobs 上。

---

## 安裝

```
/plugin marketplace add lianghsun/hsun-trainer
/plugin install hsun-trainer@hsun-trainer
```

安裝後直接用自然語言呼叫，例如：

- 「幫我用 tw-reasoning-instruct-50k 對 Gemma 3 12B 做 SFT」
- 「跑 Formosa-bench 和 TMMLU+ 評測我的模型」
- 「這個模型會輸出簡體字，用 GRPO 修掉」
- 「我要做一個懂台灣法律的模型，規劃完整 pipeline」

## 內含的 skills

| Skill | 用途 |
|---|---|
| `hsun-trainer` | 總控與路由；資料集目錄、硬體規劃、pipeline 設計 |
| `hsun-pretrain` | 繼續預訓練 (CPT / DAPT)，注入領域知識 |
| `hsun-sft` | 監督式微調：instruction / chat / reasoning / tool use |
| `hsun-grpo` | GRPO / RLVR，含**繁體中文 reward 函數庫** |
| `hsun-eval` | 用 Twinkle Eval 跑 TMMLU+ / Formosa-bench 等評測 |

## 為什麼不用 Hugging Face 官方的 skills

HF 官方的 [`huggingface/skills`](https://github.com/huggingface/skills) 很完整，
建議一起安裝。但它的 `huggingface-llm-trainer` **只做 SFT/DPO/GRPO 且只跑 HF Jobs**，
不涵蓋以下這些，正是本專案補上的部分：

| | HF 官方 | hsun-trainer |
|---|---|---|
| 繼續預訓練 (CPT) | 不支援 | 支援，含混合比例與遺忘防治 |
| 本地 / 自有多卡 | 不支援 | `accelerate` / ZeRO-3 / FSDP |
| 繁體中文評測 | 無 | TMMLU+、Formosa-bench、tw-legal-benchmark |
| 繁中 reward 函數 | 無 | 簡體字洩漏、英文漂移偵測 |
| zh-TW 資料集目錄 | 無 | 90+ 個已驗證 schema 的資料集 |

## 設計重點

**單一 recipe，兩種執行目標。** 所有訓練腳本都是自足的
[PEP 723](https://peps.python.org/pep-0723/) `uv` script，不需先安裝任何套件：

```bash
# 本地單卡
uv run scripts/train.py --config recipes/sft_gemma_zhtw.yaml

# 本地多卡（要從腳本自己的環境啟動，見下方說明）
uv sync --script scripts/train.py
"$(uv python find --script scripts/train.py)" -m accelerate.commands.launch \
    --num_processes 4 scripts/train.py --config recipes/sft_gemma_zhtw.yaml

# Hugging Face Jobs（不需自備 GPU）
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 6h \
    scripts/train.py --config recipes/sft_gemma_zhtw.yaml
```

`--config` 接受本地路徑、`https://` URL、或直接一串 JSON，
所以送上雲端時設定會跟著走，不需共用檔案系統。

**先冒煙再燒錢。** 每支訓練腳本都有 `--smoke-test`：
在 64 筆資料上跑 5 步、不存檔、不上傳，約兩分鐘內就能抓出
欄位錯誤、chat template 不合、OOM 等問題。

## 工具腳本

```bash
uv run scripts/preflight.py                    # 硬體 / 登入 / 版本檢查
uv run scripts/inspect_dataset.py <dataset>    # 欄位、格式偵測、token 長度分佈
uv run scripts/plan_memory.py --model <id> --vram 80   # full/LoRA/QLoRA 記憶體估算
uv run scripts/train.py --config <recipe>      # CPT / SFT / DPO
uv run scripts/train_grpo.py --list-rewards    # 檢視 reward 函數庫
uv run scripts/make_eval_config.py --help      # 產生 Twinkle Eval 設定
```

## 繁體中文 reward 函數

GRPO 階段可用的 reward（`uv run scripts/train_grpo.py --test-rewards` 可實測）：

| Reward | 作用 |
|---|---|
| `zhtw_purity` | 偵測簡體字洩漏，比例越高分數掉越快 |
| `no_english_drift` | 中文問題卻用英文回答時扣分（程式碼與 LaTeX 豁免） |
| `accuracy_math` | 以 `math-verify` 做符號等價比對 |
| `accuracy_mcq` | 選擇題選項字母比對 |
| `format_think` / `format_boxed` | `<think>` 與 `\boxed{}` 格式正確性 |
| `no_repetition` | 抑制 4-gram 重複迴圈 |

`zhtw_purity` 的簡體字集刻意排除兩岸通用字（里、后、台、只、干、面…），
避免懲罰正確的繁體輸出。

## 已知陷阱（已寫進 skill）

- Gemma 的 chat template **沒有** `{% generation %}`，
  `assistant_only_loss: true` 會直接報錯 →
  本專案附上輸出逐字元相同、但可正確標記 assistant 回合的修補版 template。
- 部分資料集的 `messages` 是 **JSON 字串**而非結構化 list，
  TRL 會靜默地把資料全部丟掉、用 0 筆資料「訓練成功」→ `json_columns` 解決。
- Gemma 3 4B 以上為多模態，`target_modules: all-linear` 會把 401 個 Linear
  中的 162 個（vision tower）也掛上 LoRA → recipe 內附限定語言模型的 regex。
- TRL 1.10 移除了 `warmup_ratio` 與 `GRPOConfig.max_prompt_length`，
  `scale_rewards` 從 bool 改成字串。

## 版本

對應 TRL 1.10、transformers 5.x、twinkle-eval 2.8。
腳本的 PEP 723 標頭已鎖定相容版本範圍。

## 授權

MIT。資料集與模型各自適用其原始授權 —— Gemma 系列需先於模型頁面同意授權條款。
