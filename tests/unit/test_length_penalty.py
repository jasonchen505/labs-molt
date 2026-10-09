# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hook-level tests for the DAPO overlong / ProRL stop-properly length penalties.

These drive ``RemoteExperienceMaker.compute_advantages_and_returns`` with a stub maker
and a recording estimator, i.e. they test the penalty where it actually runs — on the
merged per-rollout rewards, before the clip — not the math of an isolated helper.
"""

from types import SimpleNamespace

import pytest
import torch

import molt.trainer.rollout.experience_maker as em
from molt.cli.train_rl_ray import validate_length_penalty_args
from molt.trainer.algorithm.experience import Experience
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker


def _maker(estimator="grpo", **reward_kwargs):
    reward = {
        "clip_range": None,
        "overlong_buffer_len": None,
        "overlong_penalty_factor": 1.0,
        "stop_properly_penalty_coef": None,
    }
    reward.update(reward_kwargs)
    maker = SimpleNamespace(
        advantage_estimator=estimator,
        kl_ctl=SimpleNamespace(value=0.0),
        args=SimpleNamespace(
            reward=SimpleNamespace(**reward),
            algo=SimpleNamespace(advantage=SimpleNamespace(gamma=1.0, lam=1.0, no_whiten=True)),
            rollout=SimpleNamespace(n_samples_per_prompt=4),
            data=SimpleNamespace(max_len=2048),
        ),
    )
    maker._merge_rollout_rewards = RemoteExperienceMaker._merge_rollout_rewards.__get__(maker)
    # _per_sample_rewards is a staticmethod: assign the plain function, don't bind it.
    maker._per_sample_rewards = RemoteExperienceMaker._per_sample_rewards
    maker.compute_advantages_and_returns = RemoteExperienceMaker.compute_advantages_and_returns.__get__(maker)
    return maker


def _sample(rid, gid, reward, response_length, total_length, truncated, idx):
    return Experience(
        action_mask=torch.ones(1, 4, dtype=torch.bool),
        kl=torch.zeros(1, 4),
        rewards=torch.tensor([float(reward)]),
        response_length=torch.tensor([response_length]),
        total_length=torch.tensor([total_length]),
        truncated=torch.tensor([truncated]),
        index=[idx],
        group_ids=[gid],
        rollout_ids=[rid],
        info={"reward": torch.tensor([float(reward)])},
    )


@pytest.fixture
def record_rewards(monkeypatch):
    """Replace the estimator with a recorder; the hook's input is what we assert on."""
    captured = {}

    def fake_estimator(rewards, groups, ctx):
        captured["rewards"] = rewards.clone()
        captured["groups"] = groups
        zeros = [torch.zeros_like(m, dtype=torch.float32) for m in ctx.action_masks]
        return zeros, zeros

    monkeypatch.setattr(em, "get_advantage_estimator", lambda name: fake_estimator)
    return captured


def _overlong_maker(**kw):
    return _maker(overlong_buffer_len=200.0, overlong_penalty_factor=1.0, **kw)


def test_merge_path_penalty_is_order_invariant(record_rewards):
    # Rollout A split into two segments (same terminal reward, as multi-segment rollouts
    # carry); rollout B is a single short sample. The per-rollout penalty must not depend
    # on the order balance_experiences produced.
    # A: gen=1800, prompt=min(500,600)=500 -> total=2300 > 2048-200=1848 -> exceed=200 (capped) -> -1.0
    # B: gen=100, prompt=500 -> total=600 -> no penalty.
    seg_a1 = _sample("A", "g0", 1.0, 900, 1400, False, 0)
    seg_a2 = _sample("A", "g0", 1.0, 900, 1500, True, 1)
    samp_b = _sample("B", "g0", 0.5, 100, 600, False, 2)

    maker = _overlong_maker()
    maker.compute_advantages_and_returns([seg_a1, seg_a2, samp_b])
    first = record_rewards["rewards"]

    seg_a1 = _sample("A", "g0", 1.0, 900, 1400, False, 0)
    seg_a2 = _sample("A", "g0", 1.0, 900, 1500, True, 1)
    samp_b = _sample("B", "g0", 0.5, 100, 600, False, 2)
    maker.compute_advantages_and_returns([seg_a2, seg_a1, samp_b])
    second = record_rewards["rewards"]

    assert torch.allclose(first, torch.tensor([0.0, 0.5]))
    assert torch.allclose(second, first)


def test_penalty_applied_before_clip(record_rewards):
    # reward 1.0, overlong penalty -0.3 -> 0.7 pre-clip; clip max 0.5 -> estimator sees 0.5,
    # while info["length_penalty"] records the pre-clip -0.3.
    # total = 1908 = 1848 + 60 -> exceed=60 -> -60/200 = -0.3.
    s = _sample("A", "g0", 1.0, 1408, 1908, False, 0)
    maker = _overlong_maker(clip_range=(0.0, 0.5))
    maker.compute_advantages_and_returns([s])

    assert torch.allclose(record_rewards["rewards"], torch.tensor([0.5]))
    assert torch.allclose(s.info["length_penalty"], torch.tensor([-0.3]))
    assert torch.allclose(s.info["overlong_penalty"], torch.tensor([-0.3]))
    assert torch.allclose(s.info["stop_properly_penalty"], torch.tensor([0.0]))


def test_info_reward_stays_raw(record_rewards):
    # The raw reward signal must survive the hook untouched; the penalty lives in its
    # own metric keys.
    raw = torch.tensor([1.0])
    s = _sample("A", "g0", 1.0, 1408, 1908, False, 0)
    maker = _overlong_maker()
    maker.compute_advantages_and_returns([s])

    assert torch.allclose(s.rewards, raw)
    assert torch.allclose(s.info["reward"], raw)
    assert torch.allclose(record_rewards["rewards"], torch.tensor([0.7]))
    assert torch.allclose(s.info["length_penalty"], torch.tensor([-0.3]))


def test_stop_properly_scale_and_override(record_rewards):
    trunc = _sample("A", "g0", 1.0, 100, 600, True, 0)
    clean = _sample("B", "g0", 1.0, 100, 600, False, 1)

    maker = _maker(stop_properly_penalty_coef=0.5)
    maker.compute_advantages_and_returns([trunc, clean])
    assert torch.allclose(record_rewards["rewards"], torch.tensor([0.5, 1.0]))
    assert torch.allclose(trunc.info["stop_properly_penalty"], torch.tensor([-0.5]))
    assert torch.allclose(clean.info["stop_properly_penalty"], torch.tensor([0.0]))
    assert torch.allclose(trunc.info["length_penalty"], torch.tensor([-0.5]))

    trunc = _sample("A", "g0", 1.0, 100, 600, True, 0)
    clean = _sample("B", "g0", 1.0, 100, 600, False, 1)
    maker = _maker(stop_properly_penalty_coef=-2.0)
    maker.compute_advantages_and_returns([trunc, clean])
    assert torch.allclose(record_rewards["rewards"], torch.tensor([-2.0, 1.0]))
    assert torch.allclose(trunc.info["stop_properly_penalty"], torch.tensor([-3.0]))


def test_per_sample_path_penalty(record_rewards):
    # reinforce scores each sample independently: no merge, per-sample penalty.
    s1 = _sample("A", "g0", 1.0, 1408, 1908, False, 0)
    s2 = _sample("B", "g0", 1.0, 100, 600, False, 1)
    maker = _overlong_maker(estimator="reinforce")
    maker.compute_advantages_and_returns([s1, s2])

    assert torch.allclose(record_rewards["rewards"], torch.tensor([0.7, 1.0]))
    assert torch.allclose(s1.info["length_penalty"], torch.tensor([-0.3]))
    assert torch.allclose(s2.info["length_penalty"], torch.tensor([0.0]))


def test_no_penalty_flags_is_noop(record_rewards):
    # Flags unset: estimator sees raw rewards, no info keys added, and samples without
    # the length fields still work (backward compatible with existing callers).
    s = Experience(
        action_mask=torch.ones(1, 4, dtype=torch.bool),
        kl=torch.zeros(1, 4),
        rewards=torch.tensor([1.0]),
        index=[0],
        group_ids=["g0"],
        rollout_ids=["r0"],
        info={},
    )
    maker = _maker()
    maker.compute_advantages_and_returns([s])
    assert torch.allclose(record_rewards["rewards"], torch.tensor([1.0]))
    assert "length_penalty" not in s.info


def _penalty_args(**kw):
    base = {"overlong_buffer_len": None, "stop_properly_penalty_coef": None}
    base.update(kw)
    return SimpleNamespace(
        reward=SimpleNamespace(**base),
        data=SimpleNamespace(max_len=2048),
    )


def test_validation_rejects_bad_config():
    with pytest.raises(ValueError, match="must be positive"):
        validate_length_penalty_args(_penalty_args(overlong_buffer_len=0.0))
    with pytest.raises(ValueError, match="must be positive"):
        validate_length_penalty_args(_penalty_args(overlong_buffer_len=-10.0))
    with pytest.raises(ValueError, match="must not exceed --data.max_len"):
        validate_length_penalty_args(_penalty_args(overlong_buffer_len=4096.0))
    with pytest.raises(ValueError, match="must be <= 1"):
        validate_length_penalty_args(_penalty_args(stop_properly_penalty_coef=1.5))
    # Valid configs pass: flags unset, a sane buffer, scale and override coefs.
    validate_length_penalty_args(_penalty_args())
    validate_length_penalty_args(_penalty_args(overlong_buffer_len=200.0))
    validate_length_penalty_args(_penalty_args(stop_properly_penalty_coef=0.5))
    validate_length_penalty_args(_penalty_args(stop_properly_penalty_coef=-2.0))
