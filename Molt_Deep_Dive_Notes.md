# Molt 深入学习文档

> 对象：NVIDIA-NeMo/labs-molt —— agentic-first 的 RL 框架（~9.2K 行 RL 代码，1169⭐，Apache-2.0）
> 学习日期：2026-10-07 · 状态：只读学习，未提 PR（当前 issue 池无合适切入点，见 §9）

---

## 1. 一句话定位

Molt 是"给 agentic RL 做减法"的框架：**Ray 做 placement 和异步队列，vLLM 做 rollout，NVIDIA AutoModel + FSDP2 做训练**——三件套，纯 PyTorch/HF-native，号称同一个脚本能从 8B 跑到 1T 级 MoE（DeepSeek-V3 级别，`--fsdp.ep_size 256`）。

跟 verl/TRL/OpenRLHF 的区别：**agent-first**。在 verl 里你要去适配它的 rollout 范式；在 Molt 里 agent 就是程序本身——reward 是你写在 `Env`/`ChatAgent` 里的任意 Python（grader、多轮 tool、VLM 环境、LLM-as-judge），trainer 完全不用动。

## 2. 架构：三个盒子，一个异步循环

```
┌──────────┐   Ray async queue   ┌──────────┐   weight sync   ┌──────────────┐
│  Agent   │ ──────────────────▶ │  vLLM    │ ──────────────▶ │ Single-actor │
│ (你的程序) │   Trajectory      │ rollout  │   refit         │ Trainer      │
└──────────┘                    └──────────┘                 │ (AutoModel+  │
                                                            │  FSDP2)      │
                                                            └──────────────┘
```

代码映射（`molt/trainer/rl_trainer.py`）：
- `GenerateSamplesActor.fit()` — rollout 端：从 `rollout_queue` 取数是阻塞的，`while True` 循环直到收到 `"done"`。注释写得很实在：trainer 卡在这里说明 actor 端 idle（比如 eval 太长或 generation 成瓶颈）——这是"真正的 actor-idle 信号"。
- `TrainingActor.fit()` — 训练端：消费 queue 里的 rollout 做 `train_step`。
- `VLLMLock` — rollout 和训练之间的同步（weight sync 时要锁 vLLM）。
- `broadcast_to_vllm()` — 训练权重 refit 回 vLLM engines（`molt/trainer/fsdp/refit.py`）。

**关键洞察**：fully-async 意味着 rollout、训练、weight sync 三者 overlap。对于 DeepSeek-V3 这种大 actor，这是"不饿死"的关键——不需要 bespoke infra，Ray queue 就搞定了。

## 3. Agent 合约（最值得借鉴的部分）

`molt/agents/base.py` 定义了整个框架的"窄腰"：

```python
class Env(ABC):
    async def reset(self, state: dict) -> dict: ...      # 可选，改写初始 observation
    async def step(self, state: dict) -> Result: ...      # 核心：打分 + 可选下一轮 feedback
    async def close(self): ...                            #  teardown，必须容忍"reset 没成功就被 close"
```

- `Result`：`reward` 标量必填；可选 `observation`（多轮 feedback 文本）、`score`、`info`（数值进 metrics，字符串留在 sample 上）、`images`、`terminated`/`truncated`。
- `Runner` ABC：`StepEnvRunner`（Gym 式 step）和 `ChatAgentRunner`（OpenAI/Anthropic 兼容的 chat）都实现"一次 `execute()` 产出一个 `Trajectory`"。
- `Trajectory` dataclass：`append_action(action_tokens, action_logprobs)` / `append_feedback(...)` / `absorb_routing(...)`。

**设计评价**：这是 Gymnasium 对齐的 API，但加了 LLM 时代的东西（token 级 logprobs、multimodal、tool feedback）。`close()` 的 contract 注释（"可能在 reset 成功前被调用"）说明这是从生产环境踩坑踩出来的。对 deepseek-harness 的借鉴意义：**把"episode 生命周期"的异常路径写进 contract**，而不是靠约定。

## 4. Token-first contract：Experience

`molt/trainer/algorithm/experience.py` 的 `Experience` dataclass 是 rollout→训练的统一格式，按 RL 语义分组：

| 组 | 字段 | 形状 |
|---|---|---|
| Trajectory | `sequences` (prompt+response token ids), `attention_mask`, `action_mask` | (B, T) / (B, T-1) |
| Policy | `action_log_probs` (πθ), `base_action_log_probs` (πref), `rollout_log_probs` (πold), `routed_experts` (R3 的 top-k expert ids) | (B, T-1) |
| Optimization | `returns`, `advantages`, `values`, `kl` | (B, T-1) |
| Outcome | `rewards`, `scores`, `response_length`, `truncated`, `total_length` | (B,) |
| Metadata | `prompts`, `labels`, `images`, `index` | list |

注意 `tensor_field("step")` vs `tensor_field("episode")` 的区分，以及 `offload()`/`reload()`（CPU offload，大 actor 必备）。多轮、VLM、tool-call 的 trace 都走这一套格式——这就是 README 说的"token-first"。

## 5. 算法层：advantage estimator 注册表

`molt/trainer/algorithm/advantage.py` 用装饰器注册表，8 种 estimator 即插即用：

- `reinforce`, `reinforce_baseline`, `grpo`, `rloo`, `gae` —— 常规
- `dr_grpo` —— Dr. GRPO 变体
- `on_policy_distill` —— on-policy 蒸馏（跟 awesome-on-policy-distillation 那个方向呼应）
- **`flash_reinforce`**（2026-09 新增）：`A_i = R_i - mean(R)`，在**整个 rollout batch**上做 centering（n=1 时没有 prompt group 可平均），**不做 whitening**。binary reward 下平衡正负梯度质量；单 outcome 类别的 batch 直接 no-op。critic-free、单 rollout、号称能稳训 6000+ steps。

另有 `kl_controller.py`（Adaptive/Fixed KL）、`replay_buffer.py`（NaiveReplayBuffer）。算法层很薄——这就是"9.2K 行"的底气。

## 6. 并行与扩展：MoE 是重点

`molt/trainer/fsdp/` 下有 `strategy.py`、`packing.py`、`checkpoint.py`、`muon.py`（Muon optimizer）、`optimizer_offload.py`（Adam CPU offload）、`refit.py`。

**MoE dispatcher 选择**（`molt/models/base.py`，实测代码）：
- `MOLT_MOE_DISPATCHER` env var，默认 `hybridep`（对齐 AutoModel；intra-node NVLink 下等价于 deepep）。
- 代码注释承认：cross-node 下 hybridep/DOCA-GPUNetIO 会挂，d580 的 recipe 直接 pin deepep。
- **#90 的教训**：hybridep + `--model.gradient_checkpoint full` 会触发 activation checkpointing 的 metadata mismatch crash（shape 1296 vs 1297，差 1 个 token——跟 hybridep 要求 EP 组内 pack 相同 token 数的逻辑相关），换 deepep 绕过。
- **两个 gap**（已写成 issue 草稿，未提交）：该 env var 全仓库无文档；选了哪个 dispatcher 启动时不打日志，用户排障时不知道自己走的哪条路。

其他可调：`MOLT_GATE_PRECISION`（默认 float32，bf16 的 RMSNorm 会 non-deterministic）、`MOLT_LINEAR_BACKEND` / `MOLT_MOE_EXPERTS`、`MOLT_RMS_NORM`。共 12 个 `MOLT_*`，只有 1 个在 README 有一行文档。

## 7. CLI 与 recipes

- `molt.cli.train_sft` / `molt.cli.train_rl_ray`，arg 用 `--actor.xxx` / `--data.xxx`  dotted 风格。
- `examples/scripts/quick_start/`：5 个脚本（rl/sft × qwen3_4b/qwen3_6_35b + flash_reinforce）。
- `examples/scripts/slurm/`：大模型 recipe（glm5_2_753b、omni3_30b 等）。
- `examples/python/agents/`：`math.py`、`geo3k.py`、`chat_minimal.py` 是最好的入门读物。

## 8. 跟我工作的关联

1. **deepseek-harness**：Molt 的 `Env`/`Runner`/`Trajectory` 三件套是更干净的抽象。特别是 `close()` 的异常路径 contract 和 `Result.info` 的"数值进 metrics、字符串留 sample"设计，可以直接借鉴。
2. **vLLM serving**：`molt/trainer/rollout/router.py` 的 `VllmRouterActor`（consistent_hash 策略）是多 engine 路由的参考实现。
3. **RL 训练**：fully-async 的 queue 架构（`GenerateSamplesActor` ↔ `TrainingActor`）是解决"大 actor 饿死"问题的标准答案，比同步 rollout 简单。
4. **FlashREINFORCE**：critic-free 的新范式，值得跟进（跟我 Scholar 筛选标准里的"能否 scaling"对得上）。

## 9. 贡献面评估（为什么现在不做 PR）

- Issue 池（18 个，真实 issue 8 个）：#90 需 B200 复现且 maintainer 回避；#95 已有 PR #96；#104 是厂商推销；#101 空；#146 spam；#59/#10 是大 roadmap。
- 代码扫描（55 个文件）：几乎无 TODO；38 个测试文件；timeout/bare except/死循环等坑全干净；pre-commit + DCO 齐全。
- **结论**：当前无符合标准的 PR 切入点。已写 1 个 issue 草稿（dispatcher 文档化，`issue-draft-*.md` 同目录，未提交——判重风险，听你决定）。
- **Watch 点**：新 issue（框架还在快速成熟）、FlashREINFORCE 的后续、LoRA/PEFT 的 refit 设计（PR #57 进行中）。

## 10. 阅读路径（建议顺序）

1. `examples/python/agents/math.py` + `molt/agents/base.py` —— 先懂 agent 合约
2. `molt/trainer/algorithm/experience.py` —— 再懂数据格式
3. `molt/trainer/algorithm/advantage.py` —— 算法注册表（重点看 `flash_reinforce`）
4. `molt/trainer/rl_trainer.py`（`GenerateSamplesActor` / `TrainingActor`）—— 异步架构
5. `molt/models/base.py`（dispatcher 部分）—— MoE 实战坑
6. 技术报告 arXiv:2607.21653

## 11. 与 OpenRLHF 的对照（2026-10-07 补充）

### 血缘：实锤的"换血"而非"分家"

- Molt 55 个文件中有 **22 个（40%）带 "Adapted from OpenRLHF" 署名头**；`kl_controller.py` 除署名外与 OpenRLHF 版逐行相同。
- OpenRLHF 的 README 明确背书 Molt 为 "New Backend…more powerful than DeepSpeed"；hijkz 两边都在提交。
- 移植策略：**数据/rollout 机器全搬**（samples_generator、experience_maker、replay_buffer、kl_controller、datasets），**训练后端全换**（DeepSpeed → AutoModel/FSDP2），**agent 合约是新增的**（OpenRLHF 没有 `Env`/`Runner`/`Trajectory`）。
- 早期"NV 员工个人项目 vs 公司项目"的 tension，已通过规范开源署名 + README 互链解决。

### OpenRLHF 被放弃了吗？没有，是分工

- 没放弃：2026-09 仍有 hijkz / Excelius / Jiang Wu 的提交，10K stars，393 个 open issue 说明社区还在。
- 但提交全是维护级（SFT 修、DeepSpeed 升版、文档）；新 R&D（FlashREINFORCE、agentic、1T MoE）全在 Molt。
- **双向流动**：FlashREINFORCE 是 Molt 首发，2026-09 被 backport 回 OpenRLHF（hijkz 亲自提交）。
- OpenRLHF 自己也在加 agentic（`openrlhf/utils/agent.py` + `examples/python/agent_func*.py`），没躺平。
- 判断：**OpenRLHF = stable/community track，Molt = R&D 前沿**。

### 对照出的新发现

1. **Molt 不是纯 critic-free**：`train_rl_ray.py:159`，`--algo.advantage.estimator=gae` 时会实例化 critic（`critic_actor.py` 是活代码）。FlashREINFORCE 是 critic-free，但 PPO+critic 的路还留着。
2. **Molt 的 `compute_eval_metrics` 已正确处理加权问题**：按 group_id 分组、防 multi-turn 重复计数、防 dropped rollout 错位——SDAR 那类 unweighted batch-mean bug 在 Molt 不存在。（另：SDAR #62 的修法，OpenRLHF 自己 2026-09 才在 #1336 修了同类问题。）
3. **Porting gap**：OpenRLHF 的 `ppo_utils/` 6 个模块中，5 个搬到了 Molt，唯独 **`length_penalty.py`（DAPO overlong penalty + ProRL stop-properly penalty）没搬**。Experience 格式完全兼容，port 很直接；但 Molt 主打 agentic（多轮 tool-use 里 length penalty 不常用），漏掉可能是有意的——当 feature PR 需三思。
4. 两边 `.claude` + `AGENTS.md` 的 AI-dev 配置一模一样，同一个作者的手笔。

### 对贡献思路的影响

- 经对照，Molt 当前确实无高价值 PR 切入点（§9 结论维持）：代码干净、eval 聚合正确、issue 池无合适项。
- 唯一候选是 port `length_penalty.py`，但属 feature 且可能有意省略，暂不推进。
- Watch 点更新：OpenRLHF 的 bug fix（如 2026-09 的 SFT 系列修）是否会被同步到 Molt 的对应 "Adapted" 文件——目前看没有同步机制，可长期观察。

---
---
*本文档为个人学习笔记，放在自己 fork 上，不进上游。*
