#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "trl>=1.10,<2",
#   "transformers>=5.0,<6",
#   "peft>=0.17",
#   "datasets>=4.0",
#   "accelerate>=1.10",
#   "huggingface-hub>=0.34",
#   "pyyaml>=6",
#   "math-verify>=0.7",
#   "bitsandbytes>=0.45 ; platform_system == 'Linux'",
# ]
# ///
"""GRPO / RLVR trainer with a Traditional Chinese reward library.

Self-contained: the reward functions live in this file so `hf jobs uv run`
(single-file upload) works unchanged.

    # see and test rewards without training
    uv run scripts/train_grpo.py --list-rewards
    uv run scripts/train_grpo.py --test-rewards

    # train
    uv run scripts/train_grpo.py --config recipes/grpo_zhtw.yaml --smoke-test
    uv run scripts/train_grpo.py --config recipes/grpo_zhtw.yaml

GRPO needs a *verifiable* signal. If the task cannot be checked by a program,
use DPO (scripts/train.py --stage dpo) instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from typing import Any, Callable

# ==========================================================================
# Traditional Chinese detection
# ==========================================================================

# Characters used in Simplified Chinese that are NOT standard in Taiwan's
# Traditional orthography. Deliberately excludes forms valid in both scripts
# (里, 后, 台, 只, 干, 面, 余, 斗, 表, 发...none of which prove simplified text)
# so the reward does not punish correct zh-TW output.
SIMPLIFIED_ONLY = frozenset(
    "个们这那说会来国时过学对开门问题实现点样让认为东车长马鸟鱼见贝页风飞"
    "经给结红级纪约织统练线细终组绝续维综绿缘网编缩纲纳纵纷纸纹纺纽绅绍"
    "绑绒绕绘络绞继绩绪绮绳绵绸缆缓缔缚缝缠纯纱纬纠丝"
    "话语记论讲训议设访证识请读课谁调谈谢许诉试该详评词译诗谓误讯诚认订"
    "讥讨让诈诊诞询诫诱诵诸诺谅谊谋谎谐谚谜谣谤谦谨谬谭谱谴计诀"
    "财货质购费资贸赛赞赢账贷贵贫赶贝贞负贡责贤败贩贪购贯贱贴赋赌赏赐赔"
    "赖赚赠赡赵贺贼贿"
    "应与业专严丽举义乐习书买卖争亚产亲仅从仓仪优传伟伤价众体儿党军农"
    "医华单卫压厂历厅县参双变号叶吗员团园图圆场坏块坚声壮处备头夹夺奋奖"
    "妇妈孙宁宝宠审宪宫宽宾对寻导层尘届属岁岗岛岭峡币帅师带帮广庄庆库庙"
    "废异弃张弯弹归当录彻忆忧怀态怜总恋恳恶恼悦悬惊惧惨惯愤懒战户执扩扫"
    "扬扰抚抛抢护报担拟拥拦择挂挡挣挤挥损换据揽摆摇摊撑敌数旧昼显晒晓暂"
    "术机杀杂权条杨极构枪柜标栋栏树桥桨梦检楼欢欧歼残毁气汇汉汤沟沦沧泪"
    "泻泼泽洁洒浅浆浇浊测济浏浑浓涛润涧涨涩渊渐渔渗湾湿溃滚滞满滤滥滨滩"
    "灭灯灵灿炉炼炽烂烛烦烧烫热焕爱爷牵牺状犹独狭狮狱猎猪猫献环玛珑琐琼"
    "瑶电画畅疗疯痒痴皱盏盐监盖盗盘睁瞒码砖砚础硕确碍礼祷祸禄离积称稳穷"
    "窃窍窝窥竖竞笃笋笔笼筑筛筹签简篮篱类紧罗罚罢翘耸耻聋职联聪肃肠肤肾"
    "肿胀胁胆胜胶脏脑脸腊腻腾舆舰舱艰艳艺节芜芦苇苍苏苹茎茧荐荡荣荫药莱"
    "莲获莹莺萝萤营萧萨葱蒋蓝蔼蕴虏虑虫虽虾蚀蚁蚂蚕蛮蜗蜡蝇蝉衔补袜袭装"
    "裤观规觅视览觉触誉贝赃"
    "车轧轨轩转轮软轰轴轻载轿较辅辆辈辉辐辑输辕辖辙辞辩辫边辽达迁迈运还"
    "进远违连迟迹适选逊递逻遗邓邮邻郑酱酿释鉴"
    "针钉钓钝钞钟钠钢钩钦钧钮钱钳钻铁铃铅铜铝铭银铲铸铺链销锁锄锅锈锋锐"
    "错锚锣锤锥锦键锯锻镀镇镜"
    "闪闭闯闰闲间闷闸闹闺闻阀阁阅阎阐阔队阳阴阵阶际陆陈陕险随隐隶难雏雾"
    "静韧韩顶顷项顺须顽顾顿颂预领颇颈频颓颗颜额颠颤飘饥饭饮饰饱饲饺饼饿"
    "馆馒驭驯驰驱驳驴驶驹驻驼驾驿骂骄骆骇验骏骑骗骚骡骤鲁鲍鲜鲤鲨鲸鳄鳍"
    "鳗鳞鸠鸡鸣鸥鸦鸭鸯鸳鸽鸿鹃鹅鹉鹊鹏鹤鹦鹰麦齐齿龄龙龚龟"
)

CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN_WORD = re.compile(r"\b[A-Za-z]{3,}\b")
CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`]*`")
LATEX = re.compile(r"\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def strip_code(text: str) -> str:
    """Remove code and LaTeX so language checks do not punish legitimate ASCII."""
    for pat in (CODE_BLOCK, INLINE_CODE, LATEX):
        text = pat.sub(" ", text)
    return text


def as_text(completion: Any) -> str:
    """Normalize a completion (string, or conversational message list) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(
            str(m.get("content", "")) for m in completion if isinstance(m, dict)
        )
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def simplified_ratio(text: str) -> tuple[int, int]:
    """(simplified-only chars, total CJK chars)."""
    cjk = CJK.findall(text)
    if not cjk:
        return 0, 0
    return sum(1 for c in cjk if c in SIMPLIFIED_ONLY), len(cjk)


# ==========================================================================
# reward functions
# ==========================================================================
# Signature: f(completions, **kwargs) -> list[float | None]
# kwargs carries `prompts`, every dataset column (e.g. `ground_truth`), and
# TRL bookkeeping. Returning None for a sample excludes it from that reward.

REWARDS: dict[str, Callable] = {}
DOCS: dict[str, str] = {}


def reward(name: str, doc: str):
    def deco(fn):
        REWARDS[name] = fn
        DOCS[name] = doc
        return fn

    return deco


@reward("zhtw_purity", "1.0 for pure Traditional Chinese; scales down with simplified-char ratio.")
def zhtw_purity(completions, **kwargs):
    out = []
    for c in completions:
        text = as_text(c)
        simp, total = simplified_ratio(text)
        if total == 0:
            out.append(None)  # no CJK -> this reward does not apply
            continue
        ratio = simp / total
        # Steep: even 1% simplified is a visible defect for a zh-TW model.
        out.append(max(0.0, 1.0 - 20.0 * ratio))
    return out


@reward("no_english_drift", "Penalizes English prose leaking into a Chinese answer (code/LaTeX exempt).")
def no_english_drift(completions, prompts=None, **kwargs):
    out = []
    for i, c in enumerate(completions):
        # Only applies when a Chinese answer was actually asked for.
        if prompts is not None and i < len(prompts):
            if len(CJK.findall(as_text(prompts[i]))) < 5:
                out.append(None)
                continue
        text = strip_code(as_text(c))
        cjk = len(CJK.findall(text))
        latin = len(LATIN_WORD.findall(text))
        if cjk == 0 and latin == 0:
            out.append(None)
        elif cjk == 0:
            out.append(0.0)  # answered a Chinese prompt entirely in English
        else:
            # Allow a few technical terms; punish paragraph-level drift.
            out.append(max(0.0, 1.0 - max(0, latin - 3) / 20.0))
    return out


@reward("format_think", "Exactly one non-empty <think>...</think> block, followed by an answer.")
def format_think(completions, **kwargs):
    out = []
    for c in completions:
        text = as_text(c)
        blocks = THINK.findall(text)
        if len(blocks) != 1:
            out.append(0.0)
            continue
        inner = blocks[0].strip()
        after = THINK.sub("", text).strip()
        out.append(1.0 if len(inner) >= 20 and len(after) >= 1 else 0.3)
    return out


@reward("format_boxed", "Exactly one \\boxed{...} containing a non-empty final answer.")
def format_boxed(completions, **kwargs):
    out = []
    for c in completions:
        found = BOXED.findall(as_text(c))
        if len(found) == 1 and found[0].strip():
            out.append(1.0)
        elif len(found) > 1:
            out.append(0.2)
        else:
            out.append(0.0)
    return out


def _extract_answer(text: str) -> str | None:
    found = BOXED.findall(text)
    if found:
        return found[-1].strip()
    tail = THINK.sub("", text).strip()
    return tail.splitlines()[-1].strip() if tail else None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return re.sub(r"[\s,\u3001\u3002\uff0c\uff0e.]+$", "", s)


@reward("accuracy_mcq", "1.0 when the boxed letter matches `ground_truth` (A-J). Needs ground_truth.")
def accuracy_mcq(completions, ground_truth=None, **kwargs):
    if ground_truth is None:
        return [None] * len(completions)
    out = []
    for c, gt in zip(completions, ground_truth):
        pred = _extract_answer(as_text(c))
        if not pred:
            out.append(0.0)
            continue
        m = re.search(r"\b([A-J])\b", pred.upper())
        letter = m.group(1) if m else pred.strip().upper()[:1]
        out.append(1.0 if letter == str(gt).strip().upper()[:1] else 0.0)
    return out


@reward("accuracy_math", "Symbolic math equivalence via math-verify, with exact-match fallback.")
def accuracy_math(completions, ground_truth=None, **kwargs):
    if ground_truth is None:
        return [None] * len(completions)
    try:
        from math_verify import parse, verify

        have_mv = True
    except ImportError:
        have_mv = False

    out = []
    for c, gt in zip(completions, ground_truth):
        text = as_text(c)
        pred = _extract_answer(text)
        if not pred:
            out.append(0.0)
            continue
        gt_s = str(gt)
        if have_mv:
            try:
                if verify(parse(f"${gt_s}$"), parse(f"${pred}$")):
                    out.append(1.0)
                    continue
            except Exception:  # noqa: BLE001 - fall through to string compare
                pass
        out.append(1.0 if _norm(pred) == _norm(gt_s) else 0.0)
    return out


@reward("accuracy_exact", "Normalized exact string match against `ground_truth`.")
def accuracy_exact(completions, ground_truth=None, **kwargs):
    if ground_truth is None:
        return [None] * len(completions)
    out = []
    for c, gt in zip(completions, ground_truth):
        pred = _extract_answer(as_text(c))
        out.append(1.0 if pred and _norm(pred) == _norm(str(gt)) else 0.0)
    return out


@reward("no_repetition", "Penalizes degenerate 4-gram repetition loops.")
def no_repetition(completions, **kwargs):
    out = []
    for c in completions:
        text = as_text(c)
        units = CJK.findall(text) or text.split()
        if len(units) < 20:
            out.append(None)
            continue
        grams = [tuple(units[i : i + 4]) for i in range(len(units) - 3)]
        counts = Counter(grams)
        repeated = sum(n - 1 for n in counts.values() if n > 1)
        out.append(max(0.0, 1.0 - repeated / len(grams)))
    return out


@reward("length_target", "Gaussian bonus around `target_length` chars (set via rewards config).")
def length_target(completions, **kwargs):
    target = float(kwargs.get("_target_length") or 600)
    out = []
    for c in completions:
        n = len(as_text(c))
        out.append(math.exp(-((n - target) ** 2) / (2 * (target * 0.6) ** 2)))
    return out


# ==========================================================================
# dataset preparation -> prompt-only format
# ==========================================================================

DEFAULT_SYSTEM = (
    "你是一位以臺灣正體中文回答的助理。請先在 <think> 與 </think> 之間逐步推理，"
    "接著給出最終答案，並將最終答案放入 \\boxed{} 之中。全程只使用繁體中文。"
)


def prepare_dataset(cfg: dict, smoke: bool):
    from datasets import load_dataset

    g = cfg["grpo"]
    src = cfg["dataset"]["sources"][0]
    split = src.get("split", "train")
    cap = 32 if smoke else src.get("max_samples")
    if cap:
        split = f"{split}[:{int(cap)}]"

    print(f"    loading {src['path']} (config={src.get('config')}, split={split})")
    ds = load_dataset(src["path"], src.get("config"), split=split)
    print(f"      -> {len(ds):,} rows, columns={ds.column_names}")

    kind = g.get("dataset_kind", "chat")
    system = g.get("system_prompt", DEFAULT_SYSTEM)
    q_field = g.get("question_field", "question")
    gt_field = g.get("ground_truth_field", "answer")

    if kind == "mcq":
        letters = g.get("choice_fields", ["A", "B", "C", "D"])
        missing = [c for c in [q_field, gt_field, *letters] if c not in ds.column_names]
        if missing:
            raise SystemExit(f"[x] mcq prep needs columns {missing}; have {ds.column_names}")

        def to_prompt(row):
            opts = "\n".join(f"{L}. {row[L]}" for L in letters if row.get(L) is not None)
            user = f"{row[q_field]}\n\n{opts}\n\n請選出正確選項的英文字母。"
            return {
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "ground_truth": str(row[gt_field]).strip(),
            }

    elif kind == "math":
        def to_prompt(row):
            gt = row[gt_field]
            boxed = BOXED.findall(str(gt))
            return {
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": str(row[q_field])},
                ],
                # Many datasets put a full worked solution in the answer field;
                # the verifiable target is what sits inside \boxed{}. Rows with
                # no \boxed{} are dropped below rather than silently turning the
                # whole solution text into the target.
                "ground_truth": (boxed[-1].strip() if boxed else ""),
            }

    else:  # chat
        def to_prompt(row):
            return {
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": str(row[q_field])},
                ],
                "ground_truth": str(row.get(gt_field, "")).strip(),
            }

    ds = ds.map(to_prompt, remove_columns=ds.column_names, desc="building prompts")

    # A prompt whose target cannot be verified earns the same reward for every
    # generation, so its group advantage is zero: it consumes num_generations
    # rollouts and teaches nothing. Drop these instead of paying for them.
    before = len(ds)
    ds = ds.filter(lambda r: bool(r["ground_truth"]), desc="dropping unverifiable rows")
    if len(ds) < before:
        print(f"    dropped {before - len(ds):,}/{before:,} rows with no extractable "
              f"ground_truth ({gt_field} had no \\boxed{{}})")
    if len(ds) == 0:
        raise SystemExit(
            f"[x] No verifiable rows left. Check grpo.ground_truth_field "
            f"({gt_field!r}) and grpo.dataset_kind ({kind!r})."
        )
    print(f"    example prompt: {ds[0]['prompt'][-1]['content'][:160]}...")
    print(f"    example ground_truth: {ds[0]['ground_truth'][:80]!r}")
    return ds


# ==========================================================================
# config / training
# ==========================================================================


def load_config(spec: str) -> dict:
    import yaml

    text = spec.strip()
    if text.startswith("{"):
        return json.loads(text)
    if text.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(text, timeout=60) as r:  # noqa: S310
            return yaml.safe_load(r.read().decode("utf-8"))
    with open(text, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_rewards(cfg: dict) -> tuple[list[Callable], list[float]]:
    spec = cfg["grpo"].get("rewards") or [{"name": "format_boxed", "weight": 1.0}]
    funcs, weights = [], []
    for item in spec:
        name = item["name"] if isinstance(item, dict) else str(item)
        weight = float(item.get("weight", 1.0)) if isinstance(item, dict) else 1.0
        if name not in REWARDS:
            raise SystemExit(
                f"[x] unknown reward {name!r}. Available: {', '.join(sorted(REWARDS))}"
            )
        fn = REWARDS[name]
        if isinstance(item, dict) and "target_length" in item:
            target = item["target_length"]
            base = fn

            def fn(completions, _base=base, _t=target, **kw):  # noqa: ANN001
                return _base(completions, _target_length=_t, **kw)

            fn.__name__ = name
        funcs.append(fn)
        weights.append(weight)
        print(f"    reward {name:<18} weight={weight}")
    return funcs, weights



def assert_cuda_usable() -> None:
    """Abort if a GPU exists but torch cannot use it.

    uv resolves the newest torch, whose CUDA build can outrun the installed
    driver (e.g. torch cu130 on a 555 driver capping at CUDA 12.5). torch then
    reports cuda.is_available() == False and Trainer quietly falls back to CPU:
    the run completes, the loss curve looks plausible, and it took 100x longer
    than it should have. Fail loudly instead.
    """
    import torch

    if torch.cuda.is_available():
        return
    if os.environ.get("HSUN_ALLOW_CPU"):
        print("    [!] HSUN_ALLOW_CPU set - training on CPU on purpose")
        return
    if not sh_has_nvidia_gpu():
        return  # genuinely no GPU (macOS, CPU box) - nothing to warn about

    driver_cuda = sh_driver_cuda() or "?"
    base_ver = torch.__version__.split("+")[0]
    raise SystemExit(
        "[x] nvidia-smi reports a GPU, but torch.cuda.is_available() is False.\n"
        f"    torch {torch.__version__} was built for CUDA {torch.version.cuda}; "
        f"this driver supports up to CUDA {driver_cuda}.\n"
        "    Training would silently run on CPU.\n"
        "\n"
        "    Install a torch built for a CUDA your driver supports. Index flags on\n"
        "    `uv run`/`uv sync` do NOT work here - PyPI keeps winning the resolve -\n"
        "    so patch the script's own environment instead:\n"
        "      uv pip install --python \"$(uv python find --script scripts/train_grpo.py)\" \\\n"
        "        --reinstall-package torch \\\n"
        "        --index-url https://download.pytorch.org/whl/cu126 \\\n"
        f"        'torch=={base_ver}+cu126'\n"
        "    then run via that interpreter:\n"
        "      \"$(uv python find --script scripts/train_grpo.py)\" scripts/train_grpo.py --config <recipe>\n"
        "\n"
        "    Pick the cu tag from `nvidia-smi` (CUDA 12.x -> cu126, 13.x -> cu130).\n"
        "    Upgrading the driver is the real fix. HSUN_ALLOW_CPU=1 forces CPU."
    )


def sh_has_nvidia_gpu() -> bool:
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and out.stdout.strip().startswith("GPU ")


def sh_driver_cuda() -> str | None:
    import re as _re
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    m = _re.search(r"CUDA Version:\s*([0-9.]+)", out.stdout)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description="GRPO trainer with zh-TW rewards")
    ap.add_argument("--config", help="YAML path, https URL, or raw JSON")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--list-rewards", action="store_true")
    ap.add_argument("--test-rewards", action="store_true")
    args = ap.parse_args()

    if args.list_rewards:
        print("Available reward functions\n" + "=" * 70)
        for name in sorted(REWARDS):
            print(f"  {name:<18} {DOCS[name]}")
        print("\nCompose them in the recipe under grpo.rewards with weights.")
        return 0

    if args.test_rewards:
        samples = [
            "<think>先算 420*600=252000</think>答案是 \\boxed{11 \\frac{2}{3}}",
            "<think>这个问题需要计算</think>答案是 \\boxed{11}",   # simplified
            "The answer is definitely forty two because reasons apply here.",
            "好的好的好的好的好的好的好的好的好的好的好的好的好的好的好的好的",
        ]
        print(f"{'reward':<18}" + "".join(f"s{i:<9}" for i in range(len(samples))))
        print("-" * 70)
        for name in sorted(REWARDS):
            vals = REWARDS[name](samples, ground_truth=["11 \\frac{2}{3}"] * len(samples))
            cells = "".join(f"{('  n/a' if v is None else f'{v:5.2f}'):<10}" for v in vals)
            print(f"{name:<18}{cells}")
        print("\ns0 zh-TW correct | s1 simplified chars | s2 English drift | s3 repetition loop")
        return 0

    if not args.config:
        ap.error("--config is required unless --list-rewards/--test-rewards")

    cfg = load_config(args.config)
    print("=" * 70)
    print(f"  hsun-trainer :: GRPO{'  [SMOKE TEST]' if args.smoke_test else ''}")
    print("=" * 70)

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    print("[1/4] dataset")
    ds = prepare_dataset(cfg, args.smoke_test)

    print("[2/4] rewards")
    reward_funcs, reward_weights = build_rewards(cfg)

    print("[3/4] model")
    assert_cuda_usable()
    mc, t, g = cfg["model"], cfg["train"], cfg["grpo"]
    dtype = getattr(torch, mc.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        mc["name_or_path"],
        dtype=dtype,
        attn_implementation=mc.get("attn_implementation", "sdpa"),
        trust_remote_code=mc.get("trust_remote_code", False),
    )
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(mc["name_or_path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    peft_cfg = None
    if cfg.get("tuning", {}).get("method", "lora").lower() != "full":
        lc = cfg["tuning"]["lora"]
        peft_cfg = LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM",
        )

    print("[4/4] trainer")
    hub = cfg.get("hub", {})
    targs = GRPOConfig(
        output_dir=t["output_dir"],
        num_generations=g.get("num_generations", 8),
        max_completion_length=g.get("max_completion_length", 1024),
        temperature=g.get("temperature", 1.0),
        top_p=g.get("top_p", 1.0),
        beta=g.get("beta", 0.0),
        loss_type=g.get("loss_type", "dapo"),
        scale_rewards=g.get("scale_rewards", "group"),
        num_iterations=g.get("num_iterations", 1),
        reward_weights=reward_weights,
        mask_truncated_completions=g.get("mask_truncated_completions", True),
        use_vllm=g.get("use_vllm", False),
        vllm_mode=g.get("vllm_mode", "colocate"),
        vllm_gpu_memory_utilization=g.get("vllm_gpu_memory_utilization", 0.3),
        log_completions=True,
        num_completions_to_print=2,
        # standard knobs
        num_train_epochs=t.get("num_train_epochs", 1),
        max_steps=5 if args.smoke_test else t.get("max_steps", -1),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=1 if args.smoke_test else t.get("gradient_accumulation_steps", 4),
        learning_rate=t.get("learning_rate", 1.0e-6),
        lr_scheduler_type=t.get("lr_scheduler_type", "constant_with_warmup"),
        warmup_steps=0 if args.smoke_test else t.get("warmup_steps", 10),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        logging_steps=1 if args.smoke_test else t.get("logging_steps", 1),
        save_strategy="no" if args.smoke_test else t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 100),
        seed=t.get("seed", 42),
        report_to=[] if args.smoke_test else (t.get("report_to") or []),
        push_to_hub=False if args.smoke_test else hub.get("push_to_hub", False),
        **({"hub_model_id": hub["hub_model_id"]} if hub.get("hub_model_id") and not args.smoke_test else {}),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=targs,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    result = trainer.train()
    print(f"    loss={result.training_loss:.4f}  steps={result.global_step}")

    if args.smoke_test:
        print("\n[ok] Smoke test passed. Re-run without --smoke-test.")
        return 0

    trainer.save_model(t["output_dir"])
    tok.save_pretrained(t["output_dir"])
    print(f"    saved to {t['output_dir']}")
    if hub.get("push_to_hub"):
        trainer.push_to_hub()
    return 0


if __name__ == "__main__":
    sys.exit(main())
