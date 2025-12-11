import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class SimpleEM4EvidenceH_Refine:
    """
    三类根因 δ ∈ {0=EnvFail, 1=AgentRetryFail, 2=AgentReasoningFail}
    step 级证据: E1_gui, E2_code, E4_noresp
    case 级证据: agent_testcase_score_x (记为 C), 人工标签 delta_label (0/1/2)

    特点:
      - 半监督: 有 delta_label 的 case 做硬锚
      - C 通道 case-level 更新 ψ = P(C=1 | δ)
      - 支持各通道权重
      - 反思/ρ 留接口, 默认可以不启用 (你现在 M_reflect=1)

    用法:
      em = SimpleEM4EvidenceH_Refine(...)
      em.fit(df, col_case="test_case_id",
                 col_agent="agent_testcase_score_x",
                 col_delta="delta_label")
      post = em.predict_proba(df_new)
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 0,
        bin_thresh: float = 0.5,
        # base channel weights
        w_gui: float = 1.0,
        w_code: float = 1.0,
        w_noresp: float = 0.5,
        # agent channel
        agent_weight: float = 0.9,
        # priors
        a_pi: float = 5.0,
        b_pi: float = 5.0,  # P_delta prior
        a_c0: float = 3.0,
        b_c0: float = 3.0,  # psi EnvFail prior
        a_c1: float = 3.0,
        b_c1: float = 3.0,  # psi AgentRetryFail prior
        a_c2: float = 3.0,
        b_c2: float = 3.0,  # psi AgentReasoningFail prior
        theta_floor: float = 0.05,
        theta_ceil: float = 0.95,
        pi_floor: float = 0.02,
        temp: float = 0.8,
        # online learning parameters
        learning_rate: float = 0.1,
        decay_rate: float = 0.99,
        min_learning_rate: float = 0.01,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.bin_thresh = float(bin_thresh)

        self.w_gui = float(w_gui)
        self.w_code = float(w_code)
        self.w_no = float(w_noresp)

        self.agent_weight = float(agent_weight)

        self.a_pi, self.b_pi = float(a_pi), float(b_pi)
        self.a_c0, self.b_c0 = float(a_c0), float(b_c0)
        self.a_c1, self.b_c1 = float(a_c1), float(b_c1)
        self.a_c2, self.b_c2 = float(a_c2), float(b_c2)

        self.theta_floor = float(theta_floor)
        self.theta_ceil = float(theta_ceil)
        self.pi_floor = float(pi_floor)
        self.temp = float(temp)

        # Online learning parameters
        self.learning_rate = float(learning_rate)
        self.decay_rate = float(decay_rate)
        self.min_learning_rate = float(min_learning_rate)
        self._current_lr = float(learning_rate)

        # parameters
        # [EnvFail, AgentRetryFail, AgentReasoningFail]
        self.p_delta = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        # gui/code/noresp for 3 classes
        self.theta = np.full((3, 3), 0.5, dtype=float)
        # P(C=1|EnvFail), P(C=1|AgentRetryFail), P(C=1|AgentReasoningFail)
        self.psi = np.array([0.5, 0.5, 0.5], dtype=float)

        # Online learning: 累积统计量 (sufficient statistics)
        # 用于 incremental update
        self._n_samples_seen = 0
        self._n_cases_seen = 0
        # 累积的责任加权计数 for p_delta
        self._Nk_cumsum = np.zeros(3, dtype=float)
        # 累积的 theta 统计量: ones[d,j] 和 total[d,j]
        self._theta_ones = np.zeros((3, 3), dtype=float)
        self._theta_total = np.zeros((3, 3), dtype=float)
        # 累积的 psi 统计量
        self._psi_ones = np.zeros(3, dtype=float)
        self._psi_total = np.zeros(3, dtype=float)
        # 是否已初始化（第一次 fit 或 partial_fit）
        self._is_initialized = False

        # columns (fit 时覆盖)
        self.col_gui = "E1_gui"
        self.col_code = "E2_code"
        self.col_noresp = "E4_noresp"
        self.col_w = "weight"
        self.col_case = "test_case_id"
        self.col_agent = "agent_testcase_score_x"
        self.col_delta = "delta_label"

    # ---------- utils ----------

    @staticmethod
    def _binarize(x, thresh):
        x = np.asarray(x, float)
        x = np.clip(x, 0.0, 1.0)
        uniq = np.unique(x[~np.isnan(x)])
        if set(uniq).issubset({0.0, 1.0}):
            return x
        return (x >= thresh).astype(float)

    def _extract(self, df: pd.DataFrame):
        # ----- 证据 -----
        Eg = self._binarize(df[self.col_gui], self.bin_thresh)
        Ec = self._binarize(df[self.col_code], self.bin_thresh)

        if self.col_noresp in df.columns:
            En = self._binarize(df[self.col_noresp], self.bin_thresh)
        else:
            En = np.zeros_like(Eg)

        # E[:,0]=gui, E[:,1]=code, E[:,2]=noresp
        E = np.stack([Eg, Ec, En], axis=1)

        # ----- mask: 1 = 忽略该通道 -----
        def _get_mask(col_name):
            if col_name in df.columns:
                m = df[col_name].to_numpy().astype(float)
                return np.where(m >= 0.5, 1.0, 0.0)
            else:
                return np.zeros(Eg.shape[0], float)

        Mg = _get_mask("M_gui")
        Mc = _get_mask("M_code")
        Mn = _get_mask("M_noresp")

        M = np.stack([Mg, Mc, Mn], axis=1)

        # ----- sample 权重 -----
        if self.col_w in df.columns:
            w = np.clip(df[self.col_w].to_numpy().astype(float), 0.0, 10.0)
        else:
            w = np.ones(E.shape[0], float)

        # ----- case id -----
        case_ids = df[self.col_case].astype(str).to_numpy()

        # ----- agent_testcase_score: 行级原始值 (后面按 case 聚合) -----
        C_raw = None
        if self.col_agent in df.columns:
            arr = df[self.col_agent].to_numpy()
            C_raw = np.array([np.nan if pd.isna(v) else float(v) for v in arr])
            # 压到 {0,1} 或 NaN
            C_raw = np.where(np.isnan(C_raw), np.nan, (C_raw >= 0.5).astype(float))

        # ----- delta_label (半监督 GT，可选) -----
        delta_sup_raw = None
        if self.col_delta in df.columns:
            arr = df[self.col_delta].to_numpy()
            tmp = np.full(len(arr), np.nan)
            for i, v in enumerate(arr):
                if pd.isna(v):
                    continue
                if v in (0, 1, 2):
                    tmp[i] = int(v)
                elif isinstance(v, str):
                    vs = v.strip().lower()
                    if vs.startswith("env"):
                        tmp[i] = 0
                    elif "retry" in vs or vs.startswith("agentretry"):
                        tmp[i] = 1
                    elif "reasoning" in vs or vs.startswith("agentreasoning"):
                        tmp[i] = 2
                    elif vs.startswith("agent"):
                        # 默认 agent 相关归为 1 (AgentRetryFail)
                        tmp[i] = 1
            delta_sup_raw = tmp

        # 关键：现在返回 6 个量，包含 M
        return E, w, M, case_ids, C_raw, delta_sup_raw

    def _init_params(self, E):
        self.p_delta[:] = np.array([1 / 3, 1 / 3, 1 / 3])
        m = E.mean(axis=0)
        # EnvFail: 较低的错误率
        self.theta[0, :] = np.clip(m, 0.2, 0.8)  # EnvFail
        # AgentRetryFail: 中等错误率
        self.theta[1, :] = np.clip(m + 0.1, 0.2, 0.9)  # AgentRetryFail
        # AgentReasoningFail: 较高错误率
        self.theta[2, :] = np.clip(m + 0.2, 0.2, 0.9)  # AgentReasoningFail
        self.psi[:] = np.array([0.4, 0.6, 0.8])

    # ---------- EM fit ----------

    def fit(
        self,
        df: pd.DataFrame,
        col_case: str = "test_case_id",
        col_agent: str = "agent_testcase_score_x",
        col_delta: str = "delta_label",
        col_gui: str = "E1_gui",
        col_code: str = "E2_code",
        col_noresp: str = "E4_noresp",
        col_w: str = "weight",
    ):
        # bind column names
        self.col_case = col_case
        self.col_agent = col_agent
        self.col_delta = col_delta
        self.col_gui = col_gui
        self.col_code = col_code
        self.col_noresp = col_noresp
        self.col_w = col_w

        # E, w, case_ids, C_raw, delta_sup_raw = self._extract(df)
        E, w, M, case_ids, C_raw, delta_sup_raw = self._extract(df)
        N = E.shape[0]
        eps = 1e-9

        # case-level index
        uniq_case, inv_case = np.unique(case_ids, return_inverse=True)
        K = len(uniq_case)

        # 聚合 case-level C, delta_sup
        C_case = np.full(K, np.nan)
        delta_case = np.full(K, np.nan)
        for k in range(K):
            idx = inv_case == k
            if C_raw is not None:
                vals = C_raw[idx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    # 这里用最后一个非 nan（或 majority 都可）
                    C_case[k] = vals[-1]
            if delta_sup_raw is not None:
                vals = delta_sup_raw[idx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    delta_case[k] = vals[-1]

        self._init_params(E)

        ll_prev = -np.inf

        for it in range(self.max_iter):
            T = max(self.temp, 1e-3)

            # ----- E-step -----
            base_log = np.zeros((N, 3))
            for d in (0, 1, 2):
                p_gui, p_cod, p_no = self.theta[d]
                lg = E[:, 0] * np.log(p_gui + eps) + (1 - E[:, 0]) * np.log(1 - p_gui + eps)
                lc = E[:, 1] * np.log(p_cod + eps) + (1 - E[:, 1]) * np.log(1 - p_cod + eps)
                ln = E[:, 2] * np.log(p_no + eps) + (1 - E[:, 2]) * np.log(1 - p_no + eps)
                base_log[:, d] = self.w_gui * lg + self.w_code * lc + self.w_no * ln

            # agent channel：按 case-level C_case 映射到每一行
            agent_log = np.zeros((N, 3))
            if C_case is not None:
                C_row = C_case[inv_case]  # case 值广播到每行
                mask = ~np.isnan(C_row)
                if mask.any():
                    C_obs = C_row[mask]
                    for d in (0, 1, 2):
                        psi_d = np.clip(self.psi[d], 1e-4, 1 - 1e-4)
                        lr = C_obs * np.log(psi_d) + (1 - C_obs) * np.log(1 - psi_d)
                        agent_log[mask, d] = self.agent_weight * lr

            log_num = (np.log(self.p_delta + eps)[None, :] + base_log + agent_log) / T

            m = log_num.max(axis=1, keepdims=True)
            log_den = m + np.log(np.exp(log_num - m).sum(axis=1, keepdims=True) + eps)
            resp = np.exp(log_num - log_den)  # (N,3)

            # ----- 半监督硬锚 (case-level delta_label) -----
            if not np.all(np.isnan(delta_case)):
                for k in range(K):
                    if np.isnan(delta_case[k]):
                        continue
                    d = int(delta_case[k])  # 0, 1, or 2
                    idx = inv_case == k
                    resp[idx, :] = 0.0
                    resp[idx, d] = 1.0

            # ----- M-step -----

            # (1) π with Dirichlet prior (for 3 classes)
            Nk = (w[:, None] * resp).sum(axis=0)
            # 使用 Dirichlet 先验，简化为每个类别加相同的伪计数
            alpha_prior = np.array([self.a_pi - 1, self.a_pi - 1, self.a_pi - 1])
            self.p_delta = (Nk + alpha_prior) / (Nk.sum() + alpha_prior.sum() + eps)
            # 归一化并裁剪
            self.p_delta = self.p_delta / (self.p_delta.sum() + eps)
            self.p_delta = np.clip(self.p_delta, self.pi_floor, 1.0 - self.pi_floor)
            self.p_delta = self.p_delta / (self.p_delta.sum() + eps)  # 重新归一化

            # (2) θ for gui/code/noresp
            for d in (0, 1, 2):
                wk = w * resp[:, d]
                for j in range(3):
                    ones = (wk * E[:, j]).sum()
                    den = wk.sum()
                    if den <= 0:
                        p = 0.5
                    else:
                        # 加一点伪计数偏向 0.5，避免极端
                        num_hat = ones + 0.5
                        den_hat = den + 1.0
                        p = num_hat / (den_hat + eps)
                    self.theta[d, j] = float(np.clip(p, self.theta_floor, self.theta_ceil))

            # (3) 更新 psi (C 通道)，在 case-level 上
            if C_case is not None:
                # 对每个 case 计算该 case 属于 δ 的责任
                Rcase = np.zeros((K, 3))
                for k in range(K):
                    idx = inv_case == k
                    if not idx.any():
                        continue
                    wk = w[idx][:, None] * resp[idx, :]
                    Rcase[k, :] = wk.sum(axis=0)
                # 对三个类别分别更新
                ones_list = [0.0, 0.0, 0.0]
                den_list = [0.0, 0.0, 0.0]
                for k in range(K):
                    if np.isnan(C_case[k]):
                        continue
                    c = C_case[k]
                    r0, r1, r2 = Rcase[k, 0], Rcase[k, 1], Rcase[k, 2]
                    ones_list[0] += r0 * c
                    den_list[0] += r0
                    ones_list[1] += r1 * c
                    den_list[1] += r1
                    ones_list[2] += r2 * c
                    den_list[2] += r2
                # 更新 psi[0] (EnvFail)
                if den_list[0] > 0:
                    self.psi[0] = float(np.clip((ones_list[0] + self.a_c0 - 1) / (den_list[0] + self.a_c0 + self.b_c0 - 2 + eps), 0.02, 0.98))
                # 更新 psi[1] (AgentRetryFail)
                if den_list[1] > 0:
                    self.psi[1] = float(np.clip((ones_list[1] + self.a_c1 - 1) / (den_list[1] + self.a_c1 + self.b_c1 - 2 + eps), 0.02, 0.98))
                # 更新 psi[2] (AgentReasoningFail)
                if den_list[2] > 0:
                    # self.psi[2] = float(
                    #     np.clip((ones_list[2] + self.a_c2 - 1) /
                    #             (den_list[2] + self.a_c2 + self.b_c2 - 2 + eps),
                    #             0.02, 0.98)
                    # )
                    raw = (ones_list[2] + self.a_c2 - 1) / (den_list[2] + self.a_c2 + self.b_c2 - 2 + eps)
                    # 应用 sigmoid 正则化: sigmoid(raw - 2.0)
                    psi_sigmoid = 1.0 / (1.0 + np.exp(2.0 - raw))
                    self.psi[2] = float(np.clip(psi_sigmoid, 0.02, 0.98))

            # (4) LL convergence
            avg_ll = float((w * log_den.squeeze()).sum() / (w.sum() + eps))
            if abs(avg_ll - ll_prev) < self.tol:
                break
            ll_prev = avg_ll

        # 标记模型已初始化
        self._is_initialized = True
        self._n_samples_seen = N
        self._n_cases_seen = K

    # ---------- Online Learning ----------

    def partial_fit(
        self,
        df: pd.DataFrame,
        col_case: str = "test_case_id",
        col_agent: str = "agent_testcase_score_x",
        col_delta: str = "delta_label",
        col_gui: str = "E1_gui",
        col_code: str = "E2_code",
        col_noresp: str = "E4_noresp",
        col_w: str = "weight",
        n_iter: int = 1,
    ):
        """
        Online learning: 增量更新模型参数

        使用新收集的证据数据更新模型，不需要存储历史数据。
        适用于实时收集 gui_evidence, reflection_evidence, code_evidence 后更新模型。

        Args:
            df: 新收集的证据数据 DataFrame
            col_case: case ID 列名
            col_agent: agent 评分列名
            col_delta: delta 标签列名（可选，用于半监督）
            col_gui: GUI 证据列名
            col_code: 代码证据列名
            col_noresp: noresp 证据列名
            col_w: 权重列名
            n_iter: 每次 partial_fit 的 EM 迭代次数

        Returns:
            self: 返回自身，支持链式调用
        """
        # bind column names
        self.col_case = col_case
        self.col_agent = col_agent
        self.col_delta = col_delta
        self.col_gui = col_gui
        self.col_code = col_code
        self.col_noresp = col_noresp
        self.col_w = col_w

        E, w, M, case_ids, C_raw, delta_sup_raw = self._extract(df)
        N = E.shape[0]
        eps = 1e-9

        # 如果是第一次调用，初始化参数
        if not self._is_initialized:
            self._init_params(E)
            self._is_initialized = True

        # case-level index
        uniq_case, inv_case = np.unique(case_ids, return_inverse=True)
        K = len(uniq_case)

        # 聚合 case-level C, delta_sup
        C_case = np.full(K, np.nan)
        delta_case = np.full(K, np.nan)
        for k in range(K):
            idx = inv_case == k
            if C_raw is not None:
                vals = C_raw[idx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    C_case[k] = vals[-1]
            if delta_sup_raw is not None:
                vals = delta_sup_raw[idx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    delta_case[k] = vals[-1]

        # Online EM iterations
        for _ in range(n_iter):
            T = max(self.temp, 1e-3)

            # ----- E-step: 计算责任 -----
            base_log = np.zeros((N, 3))
            for d in (0, 1, 2):
                p_gui, p_cod, p_no = self.theta[d]
                lg = E[:, 0] * np.log(p_gui + eps) + (1 - E[:, 0]) * np.log(1 - p_gui + eps)
                lc = E[:, 1] * np.log(p_cod + eps) + (1 - E[:, 1]) * np.log(1 - p_cod + eps)
                ln = E[:, 2] * np.log(p_no + eps) + (1 - E[:, 2]) * np.log(1 - p_no + eps)

                # 应用 mask
                lg[M[:, 0] == 1] = 0.0
                lc[M[:, 1] == 1] = 0.0
                ln[M[:, 2] == 1] = 0.0

                base_log[:, d] = self.w_gui * lg + self.w_code * lc + self.w_no * ln

            # agent channel
            agent_log = np.zeros((N, 3))
            if C_case is not None:
                C_row = C_case[inv_case]
                mask = ~np.isnan(C_row)
                if mask.any():
                    C_obs = C_row[mask]
                    for d in (0, 1, 2):
                        psi_d = np.clip(self.psi[d], 1e-4, 1 - 1e-4)
                        lr = C_obs * np.log(psi_d) + (1 - C_obs) * np.log(1 - psi_d)
                        agent_log[mask, d] = self.agent_weight * lr

            log_num = (np.log(self.p_delta + eps)[None, :] + base_log + agent_log) / T

            m = log_num.max(axis=1, keepdims=True)
            log_den = m + np.log(np.exp(log_num - m).sum(axis=1, keepdims=True) + eps)
            resp = np.exp(log_num - log_den)

            # 半监督硬锚
            if not np.all(np.isnan(delta_case)):
                for k in range(K):
                    if np.isnan(delta_case[k]):
                        continue
                    d = int(delta_case[k])
                    idx = inv_case == k
                    resp[idx, :] = 0.0
                    resp[idx, d] = 1.0

            # ----- M-step: 使用学习率进行增量更新 -----
            lr = self._current_lr

            # (1) 更新 p_delta（使用指数移动平均）
            Nk_new = (w[:, None] * resp).sum(axis=0)
            alpha_prior = np.array([self.a_pi - 1, self.a_pi - 1, self.a_pi - 1])
            p_delta_new = (Nk_new + alpha_prior) / (Nk_new.sum() + alpha_prior.sum() + eps)
            p_delta_new = p_delta_new / (p_delta_new.sum() + eps)
            p_delta_new = np.clip(p_delta_new, self.pi_floor, 1.0 - self.pi_floor)
            p_delta_new = p_delta_new / (p_delta_new.sum() + eps)

            # EMA 更新
            self.p_delta = (1 - lr) * self.p_delta + lr * p_delta_new

            # (2) 更新 theta
            theta_new = np.zeros((3, 3), dtype=float)
            for d in (0, 1, 2):
                wk = w * resp[:, d]
                for j in range(3):
                    # 只计算未被 mask 的样本
                    valid_mask = M[:, j] == 0
                    ones = (wk[valid_mask] * E[valid_mask, j]).sum()
                    den = wk[valid_mask].sum()
                    if den <= 0:
                        p = self.theta[d, j]  # 保持原值
                    else:
                        num_hat = ones + 0.5
                        den_hat = den + 1.0
                        p = num_hat / (den_hat + eps)
                    theta_new[d, j] = float(np.clip(p, self.theta_floor, self.theta_ceil))

            # EMA 更新
            self.theta = (1 - lr) * self.theta + lr * theta_new

            # (3) 更新 psi
            if C_case is not None:
                Rcase = np.zeros((K, 3))
                for k in range(K):
                    idx = inv_case == k
                    if not idx.any():
                        continue
                    wk = w[idx][:, None] * resp[idx, :]
                    Rcase[k, :] = wk.sum(axis=0)

                ones_list = [0.0, 0.0, 0.0]
                den_list = [0.0, 0.0, 0.0]
                for k in range(K):
                    if np.isnan(C_case[k]):
                        continue
                    c = C_case[k]
                    r0, r1, r2 = Rcase[k, 0], Rcase[k, 1], Rcase[k, 2]
                    ones_list[0] += r0 * c
                    den_list[0] += r0
                    ones_list[1] += r1 * c
                    den_list[1] += r1
                    ones_list[2] += r2 * c
                    den_list[2] += r2

                psi_new = self.psi.copy()
                for d in range(3):
                    if den_list[d] > 0:
                        if d == 2:
                            # AgentReasoningFail: sigmoid 正则化
                            raw = (ones_list[d] + self.a_c2 - 1) / (den_list[d] + self.a_c2 + self.b_c2 - 2 + eps)
                            psi_sigmoid = 1.0 / (1.0 + np.exp(2.0 - raw))
                            psi_new[d] = float(np.clip(psi_sigmoid, 0.02, 0.98))
                        else:
                            a_c = self.a_c0 if d == 0 else self.a_c1
                            b_c = self.b_c0 if d == 0 else self.b_c1
                            psi_new[d] = float(np.clip((ones_list[d] + a_c - 1) / (den_list[d] + a_c + b_c - 2 + eps), 0.02, 0.98))

                # EMA 更新
                self.psi = (1 - lr) * self.psi + lr * psi_new

        # 更新统计量
        self._n_samples_seen += N
        self._n_cases_seen += K

        # 学习率衰减
        self._current_lr = max(self._current_lr * self.decay_rate, self.min_learning_rate)

        return self

    def online_update(
        self,
        gui_evidence: Optional[int] = None,
        code_evidence: Optional[int] = None,
        reflection_evidence: Optional[float] = None,
        agent_score: Optional[float] = None,
        test_case_id: str = "online_case",
        weight: float = 1.0,
        delta_label: Optional[int] = None,
    ):
        """
        单样本 online 更新：接收单个证据并更新模型

        这是最简单的 online learning 接口，每收集一条证据就可以调用。

        Args:
            gui_evidence: GUI 证据（0/1，None 表示无效）
            code_evidence: 代码证据（0/1，None 表示无效）
            reflection_evidence: 反思证据（可选）
            agent_score: Agent 评分（0/1）
            test_case_id: 测试用例 ID
            weight: 样本权重
            delta_label: 半监督标签（0/1/2，可选）

        Returns:
            self: 返回自身，支持链式调用
        """
        # 构建单行 DataFrame
        row = {
            self.col_case: test_case_id,
            self.col_gui: gui_evidence if gui_evidence is not None else 0,
            self.col_code: code_evidence if code_evidence is not None else 0,
            self.col_noresp: 0,
            "M_gui": 0 if gui_evidence is not None else 1,
            "M_code": 0 if code_evidence is not None else 1,
            "M_noresp": 1,
            self.col_w: weight,
        }

        if agent_score is not None:
            row[self.col_agent] = agent_score

        if delta_label is not None:
            row[self.col_delta] = delta_label

        df = pd.DataFrame([row])
        return self.partial_fit(df, n_iter=1)

    def reset_learning_rate(self, learning_rate: Optional[float] = None):
        """
        重置学习率（例如在新任务开始时）

        Args:
            learning_rate: 新的学习率，如果为 None 则使用初始学习率
        """
        if learning_rate is not None:
            self._current_lr = float(learning_rate)
        else:
            self._current_lr = self.learning_rate

    def get_online_stats(self) -> Dict[str, Any]:
        """
        获取 online learning 的统计信息

        Returns:
            包含统计信息的字典
        """
        return {
            "n_samples_seen": self._n_samples_seen,
            "n_cases_seen": self._n_cases_seen,
            "current_learning_rate": self._current_lr,
            "is_initialized": self._is_initialized,
        }

    # ---------- inference ----------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """返回 step-level P(EnvFail), P(AgentRetryFail), P(AgentReasoningFail)，使用与 fit 一致的 M_* 掩码"""
        E, w, M, case_ids, C_raw, _ = self._extract(df)
        N = E.shape[0]
        eps = 1e-9

        uniq_case, inv_case = np.unique(case_ids, return_inverse=True)
        K = len(uniq_case)

        # ---- case 级 C，与 fit 同逻辑 ----
        C_case = np.full(K, np.nan)
        if C_raw is not None:
            for k in range(K):
                idx = inv_case == k
                vals = C_raw[idx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    C_case[k] = vals[-1]

        # ---- 基于证据通道 + 掩码的 log-likelihood ----
        base_log = np.zeros((N, 3))
        for d in (0, 1, 2):
            p_gui, p_cod, p_no = self.theta[d]

            lg = E[:, 0] * np.log(p_gui + eps) + (1 - E[:, 0]) * np.log(1 - p_gui + eps)
            lc = E[:, 1] * np.log(p_cod + eps) + (1 - E[:, 1]) * np.log(1 - p_cod + eps)
            ln = E[:, 2] * np.log(p_no + eps) + (1 - E[:, 2]) * np.log(1 - p_no + eps)

            # 关键：对 mask=1 的位置，把该通道贡献清空（忽略该证据）
            lg[M[:, 0] == 1] = 0.0
            lc[M[:, 1] == 1] = 0.0
            ln[M[:, 2] == 1] = 0.0

            base_log[:, d] = self.w_gui * lg + self.w_code * lc + self.w_no * ln

        # ---- agent_testcase_score 通道（case 级）----
        agent_log = np.zeros((N, 3))
        if C_case is not None:
            C_row = C_case[inv_case]
            mask = ~np.isnan(C_row)
            if mask.any():
                C_obs = C_row[mask]
                for d in (0, 1, 2):
                    psi_d = np.clip(self.psi[d], 1e-4, 1 - 1e-4)
                    lr = C_obs * np.log(psi_d) + (1 - C_obs) * np.log(1 - psi_d)
                    agent_log[mask, d] = self.agent_weight * lr

        # ---- 合成 posterior ----
        log_num = np.log(self.p_delta + eps)[None, :] + base_log + agent_log
        m = log_num.max(axis=1, keepdims=True)
        log_den = m + np.log(np.exp(log_num - m).sum(axis=1, keepdims=True) + eps)
        post = np.exp(log_num - log_den)
        # [:,0]=P(EnvFail), [:,1]=P(AgentRetryFail), [:,2]=P(AgentReasoningFail)
        return post

    def get_params(self) -> Dict[str, Any]:
        return {
            "P_delta": {"EnvFail": float(self.p_delta[0]), "AgentRetryFail": float(self.p_delta[1]), "AgentReasoningFail": float(self.p_delta[2])},
            "theta": {
                "EnvFail": {"E_gui": float(self.theta[0, 0]), "E_code": float(self.theta[0, 1]), "E_noresp": float(self.theta[0, 2])},
                "AgentRetryFail": {"E_gui": float(self.theta[1, 0]), "E_code": float(self.theta[1, 1]), "E_noresp": float(self.theta[1, 2])},
                "AgentReasoningFail": {"E_gui": float(self.theta[2, 0]), "E_code": float(self.theta[2, 1]), "E_noresp": float(self.theta[2, 2])},
            },
            "psi": {
                "EnvFail": float(self.psi[0]),
                "AgentRetryFail": float(self.psi[1]),
                "AgentReasoningFail": float(self.psi[2]),
            },
        }

    def load_params(self, params: Dict[str, Any]):
        """
        从参数字典加载已训练的参数
        """
        self.p_delta[0] = float(params["P_delta"]["EnvFail"])
        self.p_delta[1] = float(params["P_delta"]["AgentRetryFail"])
        self.p_delta[2] = float(params["P_delta"]["AgentReasoningFail"])

        self.theta[0, 0] = float(params["theta"]["EnvFail"]["E_gui"])
        self.theta[0, 1] = float(params["theta"]["EnvFail"]["E_code"])
        self.theta[0, 2] = float(params["theta"]["EnvFail"]["E_noresp"])
        self.theta[1, 0] = float(params["theta"]["AgentRetryFail"]["E_gui"])
        self.theta[1, 1] = float(params["theta"]["AgentRetryFail"]["E_code"])
        self.theta[1, 2] = float(params["theta"]["AgentRetryFail"]["E_noresp"])
        self.theta[2, 0] = float(params["theta"]["AgentReasoningFail"]["E_gui"])
        self.theta[2, 1] = float(params["theta"]["AgentReasoningFail"]["E_code"])
        self.theta[2, 2] = float(params["theta"]["AgentReasoningFail"]["E_noresp"])

        self.psi[0] = float(params["psi"]["EnvFail"])
        self.psi[1] = float(params["psi"]["AgentRetryFail"])
        self.psi[2] = float(params["psi"]["AgentReasoningFail"])

    def _score_avg_ll(self, E, w, case_ids, C_raw):
        """
        计算平均 log-likelihood
        参数：
        - E: (N, 3) 证据矩阵 [gui, code, noresp]
        - w: (N,) 权重
        - case_ids: (N,) case ID（用于聚合 C）
        - C_raw: (N,) agent score（可能包含 NaN）
        """
        N = E.shape[0]
        eps = 1e-9

        # 聚合 C_case（和 predict_proba 同逻辑）
        uniq_case, inv_case = np.unique(case_ids, return_inverse=True)
        K = len(uniq_case)
        C_case = np.full(K, np.nan)
        if C_raw is not None:
            for k in range(K):
                idx = inv_case == k
                vals = C_raw[idx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    C_case[k] = vals[-1]

        # base log-likelihood
        base_log = np.zeros((N, 3))
        for d in (0, 1, 2):
            p_gui, p_cod, p_no = self.theta[d]
            lg = E[:, 0] * np.log(p_gui + eps) + (1 - E[:, 0]) * np.log(1 - p_gui + eps)
            lc = E[:, 1] * np.log(p_cod + eps) + (1 - E[:, 1]) * np.log(1 - p_cod + eps)
            ln = E[:, 2] * np.log(p_no + eps) + (1 - E[:, 2]) * np.log(1 - p_no + eps)
            base_log[:, d] = self.w_gui * lg + self.w_code * lc + self.w_no * ln

        # agent log-likelihood
        agent_log = np.zeros((N, 3))
        if C_case is not None:
            C_row = C_case[inv_case]
            mask = ~np.isnan(C_row)
            if mask.any():
                C_obs = C_row[mask]
                for d in (0, 1, 2):
                    psi_d = np.clip(self.psi[d], 1e-4, 1 - 1e-4)
                    lr = C_obs * np.log(psi_d) + (1 - C_obs) * np.log(1 - psi_d)
                    agent_log[mask, d] = self.agent_weight * lr

        # 计算 log-likelihood
        log_num = np.log(self.p_delta + eps)[None, :] + base_log + agent_log
        m = log_num.max(axis=1, keepdims=True)
        log_den = m + np.log(np.exp(log_num - m).sum(axis=1, keepdims=True) + eps)

        return float((w * log_den.squeeze()).sum() / (w.sum() + eps))


def analyze_flips(val_df: pd.DataFrame, out_dir: Optional[str] = None):
    """区分两类子集: 原判断正确 vs 错误，查看flip比例"""
    from pathlib import Path

    df = val_df.copy()
    df["is_correct_before"] = df["human_gt"] == df["agent_original"]
    df["is_correct_after"] = df["human_gt"] == df["corrected_label"]
    df["is_flipped"] = df["agent_original"] != df["corrected_label"]

    # subset A: 原判断正确
    subset_A = df[df["is_correct_before"]]
    misflipped = subset_A[subset_A["is_flipped"]]
    if len(subset_A) > 0:
        print(f"[Subset A] 原判断正确: {len(subset_A)} cases, " f"被误翻 {len(misflipped)} ({len(misflipped)/len(subset_A):.2%})")

        # 进一步区分原始标签是0和原始标签是1的误翻情况
        misflipped_0 = misflipped[misflipped["agent_original"] == 0]
        misflipped_1 = misflipped[misflipped["agent_original"] == 1]
        subset_A_0 = subset_A[subset_A["agent_original"] == 0]
        subset_A_1 = subset_A[subset_A["agent_original"] == 1]

        if len(subset_A_0) > 0:
            print(f"  - 原始标签=0: {len(subset_A_0)} cases, " f"被误翻 {len(misflipped_0)} ({len(misflipped_0)/len(subset_A_0):.2%})")
        if len(subset_A_1) > 0:
            print(f"  - 原始标签=1: {len(subset_A_1)} cases, " f"被误翻 {len(misflipped_1)} ({len(misflipped_1)/len(subset_A_1):.2%})")

        # 保存数据
        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # 保存被误翻的数据（按原始标签分组）
            if len(misflipped_0) > 0:
                misflipped_0.to_csv(out_path / "misflipped_original_0.csv", index=False)
                print(f"  已保存原始标签=0的误翻数据到: {out_path / 'misflipped_original_0.csv'}")
            if len(misflipped_1) > 0:
                misflipped_1.to_csv(out_path / "misflipped_original_1.csv", index=False)
                print(f"  已保存原始标签=1的误翻数据到: {out_path / 'misflipped_original_1.csv'}")

            # 保存原判断正确的所有数据（按原始标签分组）
            if len(subset_A_0) > 0:
                subset_A_0.to_csv(out_path / "subset_A_original_0.csv", index=False)
                print(f"  已保存原始标签=0的原判断正确数据到: {out_path / 'subset_A_original_0.csv'}")
            if len(subset_A_1) > 0:
                subset_A_1.to_csv(out_path / "subset_A_original_1.csv", index=False)
                print(f"  已保存原始标签=1的原判断正确数据到: {out_path / 'subset_A_original_1.csv'}")
    else:
        print(f"[Subset A] 原判断正确: {len(subset_A)} cases, 被误翻 {len(misflipped)}")

    # subset B: 原判断错误
    subset_B = df[~df["is_correct_before"]]
    corrected = subset_B[subset_B["is_correct_after"]]
    if len(subset_B) > 0:
        print(f"[Subset B] 原判断错误: {len(subset_B)} cases, " f"被成功纠正 {len(corrected)} ({len(corrected)/len(subset_B):.2%})")
    else:
        print(f"[Subset B] 原判断错误: {len(subset_B)} cases, 被成功纠正 {len(corrected)}")

    # 再打印最终 confusion
    print("\n=== Confusion Matrix (GT vs Corrected) ===")
    print(pd.crosstab(df["human_gt"], df["corrected_label"], rownames=["GT"], colnames=["Corrected"]))


def confusion_matrix(val_df: pd.DataFrame):
    """统计人类标注 vs 矫正后结果"""
    conf_matrix = pd.crosstab(val_df["human_gt"], val_df["corrected_label"], rownames=["GT"], colnames=["Corrected"])
    print("=== Confusion Matrix (GT vs Corrected) ===")
    print(conf_matrix)
    acc = (val_df["human_gt"] == val_df["corrected_label"]).mean()
    print(f"Accuracy after correction: {acc:.3f}")
    return conf_matrix


def correct_agent_judgment(
    df: pd.DataFrame,
    em: SimpleEM4EvidenceH_Refine,
    tau_agentfail: float = 0.7,
    tau_envfail: float = 0.7,
    alpha: float = 0.75,
    tau_envfail_high: float = 0.7,
    col_case: str = "test_case_id",
    col_agent: str = "agent_testcase_score_x",
):
    """
    对每个 case：
      - 汇总 step-level posterior 得到 P_case_AgentFail
      - 与 agent 原判 (C_case) 结合，给出纠偏动作
    """
    post = em.predict_proba(df)
    df_tmp = df.copy()
    df_tmp["P_EnvFail"] = post[:, 0]
    # AgentRetryFail (1) 和 AgentReasoningFail (2) 都视为 AgentFail
    df_tmp["P_AgentFail"] = post[:, 1] + post[:, 2]

    rows = []
    for cid, g in df_tmp.groupby(col_case):
        # agent 原判: 取该 case 最后一个非 nan
        C_vals = g[col_agent].dropna().values
        if cid == "web_14_2":
            print(f"C_vals: {C_vals}")
        # 1=PASS?, 0=FAIL? 按你现在的定义自行对应
        C_case = int(C_vals[-1]) if len(C_vals) else None

        gt = g["phi"].dropna().values[-1]

        # 聚合为 case-level AgentFail 概率（阻塞口径）
        q = np.clip(g["P_AgentFail"].values, 0.0, 1.0)
        # 阻塞概率: 越多高风险 step, 越趋向 AgentFail
        P_case_AgentFail = 1.0 - float(np.prod((1.0 - q) ** alpha))
        P_case_EnvFail = 1.0 - P_case_AgentFail

        action = "keep_AgentJudge"
        corrected = C_case

        if C_case is not None:
            # 场景1: agent 判 FAIL (0)，我们要区分 Env vs Agent
            if C_case == 0:
                if P_case_AgentFail >= tau_agentfail:
                    corrected = 1  # 认定 AgentFail
                    action = "flip_to_AgentFail"
                elif P_case_EnvFail >= tau_envfail:
                    corrected = 0  # 环境问题，维持
                    action = "keep_EnvFail"
                # 介于两阈值之间可以保守 keep_AgentJudge/UNK

            # 场景2: agent 判 PASS (1)，可选：当强 EnvFail 证据时 flip
            elif C_case == 1:
                # 如果你只关心纠正误报，可以先不动这里
                if P_case_EnvFail >= tau_envfail_high:  # tau_envfail:
                    corrected = 0
                    action = "flip_to_EnvFail"
        rows.append(
            dict(
                case_id=cid,
                human_gt=gt,
                agent_original=C_case,
                P_case_EnvFail=P_case_EnvFail,
                P_case_AgentFail=P_case_AgentFail,
                corrected_label=corrected,
                action=action,
            )
        )
    val_df = pd.DataFrame(rows).sort_values("case_id")
    acc_original = val_df["agent_original"] == val_df["human_gt"]
    print(f"Accuracy: {acc_original.mean()}")

    acc_correct = val_df["corrected_label"] == val_df["human_gt"]
    print(f"Accuracy: {acc_correct.mean()}")
    analyze_flips(val_df)
    confusion_matrix(val_df)
    return val_df


class EMManager:
    """
    EM 模型管理器

    整合以下功能：
    1. 加载预训练的 em_params.json 参数
    2. 收集 OS agent 运行时的证据
    3. 整理证据为 EM 格式
    4. 进行预测和纠偏

    用法示例:
    ```python
    # 初始化
    em_manager = EMManager(params_path="appeval/data/em_params.json")

    # 在 OS agent 每一步收集证据后更新
    em_manager.add_evidence(
        gui_evidence=1,
        code_evidence=0,
        agent_noresp=0,
        test_case_id="web_01",
    )

    # 获取预测结果
    prediction = em_manager.predict()

    # 进行纠偏
    corrected = em_manager.correct_judgment(agent_original=0)
    ```
    """

    # 默认参数文件路径
    DEFAULT_PARAMS_PATH = "appeval/data/em_params.json"

    def __init__(
        self,
        params_path: Optional[str] = None,
        enable_online_learning: bool = True,
        learning_rate: float = 0.1,
        tau_agentfail: float = 0.7,
        tau_envfail: float = 0.7,
        alpha: float = 0.75,
    ):
        """
        初始化 EM 管理器

        Args:
            params_path: 预训练参数文件路径，为 None 时使用默认参数
            enable_online_learning: 是否启用在线学习
            learning_rate: 在线学习的学习率
            tau_agentfail: AgentFail 判断阈值
            tau_envfail: EnvFail 判断阈值
            alpha: 纠偏时的阻塞概率指数
        """
        self.logger = logging.getLogger(f"{__name__}.EMManager")

        # 创建 EM 模型
        self.em = SimpleEM4EvidenceH_Refine(
            learning_rate=learning_rate,
        )

        # 存储证据数据
        self._evidence_rows: List[Dict] = []
        self._current_case_id: str = "default"

        # 纠偏参数
        self.tau_agentfail = tau_agentfail
        self.tau_envfail = tau_envfail
        self.alpha = alpha

        # 在线学习开关
        self.enable_online_learning = enable_online_learning

        # 加载预训练参数
        if params_path is not None:
            self.load_params(params_path)
        else:
            # 尝试加载默认参数文件
            default_path = self._find_default_params_path()
            if default_path:
                self.load_params(default_path)
                self.logger.info(f"已加载默认参数: {default_path}")

    def _find_default_params_path(self) -> Optional[str]:
        """查找默认参数文件路径"""
        # 尝试多个可能的路径
        possible_paths = [
            self.DEFAULT_PARAMS_PATH,
            os.path.join(os.path.dirname(__file__), "..", "data", "em_params.json"),
            os.path.join(os.path.dirname(__file__), "../../appeval/data/em_params.json"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def load_params(self, params_path: str):
        """
        从 JSON 文件加载预训练参数

        Args:
            params_path: 参数文件路径
        """
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"参数文件不存在: {params_path}")

        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.em.load_params(params)
        self.em._is_initialized = True
        self.logger.info(f"已加载 EM 参数: {params_path}")

    def save_params(self, params_path: str):
        """
        保存当前参数到 JSON 文件

        Args:
            params_path: 参数文件路径
        """
        params = self.em.get_params()

        # 确保目录存在
        os.makedirs(os.path.dirname(params_path), exist_ok=True)

        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

        self.logger.info(f"已保存 EM 参数: {params_path}")

    def set_case_id(self, case_id: str):
        """
        设置当前测试用例 ID

        Args:
            case_id: 测试用例 ID
        """
        self._current_case_id = case_id

    def add_evidence(
        self,
        gui_evidence: Optional[int] = None,
        code_evidence: Optional[int] = None,
        agent_noresp: Optional[int] = None,
        agent_score: Optional[float] = None,
        test_case_id: Optional[str] = None,
        step_id: Optional[str] = None,
        weight: float = 1.0,
        iter_num: int = 0,
    ):
        """
        添加一条证据数据

        Args:
            gui_evidence: GUI 证据（1=命中, 0=未命中, None=无效）
            code_evidence: 代码证据（1=已实现, 0=未实现）
            agent_noresp: Agent 无响应证据（1=有问题, 0=正常）
            agent_score: Agent 评分（0-1）
            test_case_id: 测试用例 ID
            step_id: 步骤 ID
            weight: 样本权重
            iter_num: 迭代编号
        """
        if test_case_id is None:
            test_case_id = self._current_case_id

        if step_id is None:
            step_id = f"{test_case_id}_iter_{iter_num}"

        # 处理 GUI 证据
        if gui_evidence is None or gui_evidence == -1:
            M_gui = 1
            E_gui = 0
        else:
            M_gui = 0
            E_gui = gui_evidence

        # 处理代码证据
        if code_evidence is None:
            M_code = 1
            E_code = 0
        else:
            M_code = 0
            E_code = code_evidence

        # 处理 noresp 证据
        if agent_noresp is None:
            M_noresp = 1
            E_noresp = 0
        else:
            M_noresp = 0
            E_noresp = agent_noresp

        # 构建证据行
        row = {
            "test_case_id": test_case_id,
            "step": step_id,
            "E1_gui": E_gui,
            "E2_code": E_code,
            "E4_noresp": E_noresp,
            "M_gui": M_gui,
            "M_code": M_code,
            "M_noresp": M_noresp,
            "weight": weight,
        }

        if agent_score is not None:
            row["agent_testcase_score_x"] = agent_score

        self._evidence_rows.append(row)

        # 如果启用在线学习，立即更新模型
        if self.enable_online_learning:
            self.em.online_update(
                gui_evidence=E_gui if M_gui == 0 else None,
                code_evidence=E_code if M_code == 0 else None,
                agent_score=agent_score,
                test_case_id=test_case_id,
                weight=weight,
            )

    def add_evidence_from_collector(
        self,
        evidence_collector,
        code_evidence: Optional[Dict[str, int]] = None,
        agent_score: Optional[float] = None,
    ):
        """
        从 OnlineEvidenceCollector 添加证据

        Args:
            evidence_collector: OnlineEvidenceCollector 实例
            code_evidence: 代码证据字典 {step_id: code_evidence}
            agent_score: Agent 评分
        """
        from appeval.utils.em_data_process import convert_evidences_to_em_format

        evidences = evidence_collector.get_evidences_for_em()
        project_name = evidence_collector.project_name

        df = convert_evidences_to_em_format(
            evidences=evidences,
            project_name=project_name,
            code_evidence=code_evidence,
            agent_judge={project_name: agent_score} if agent_score is not None else None,
        )

        # 添加到证据行
        for _, row in df.iterrows():
            self._evidence_rows.append(row.to_dict())

        # 如果启用在线学习，更新模型
        if self.enable_online_learning and not df.empty:
            self.em.partial_fit(df)

    def get_evidence_dataframe(self) -> pd.DataFrame:
        """
        获取当前收集的证据 DataFrame

        Returns:
            证据 DataFrame
        """
        if not self._evidence_rows:
            return pd.DataFrame()
        return pd.DataFrame(self._evidence_rows)

    def predict(self, case_id: Optional[str] = None) -> Dict[str, float]:
        """
        预测当前证据的根因概率

        Args:
            case_id: 测试用例 ID，为 None 时使用当前 case

        Returns:
            包含三类根因概率的字典
        """
        df = self.get_evidence_dataframe()

        if df.empty:
            return {
                "P_EnvFail": 1 / 3,
                "P_AgentRetryFail": 1 / 3,
                "P_AgentReasoningFail": 1 / 3,
            }

        # 过滤指定 case
        if case_id is not None:
            df = df[df["test_case_id"] == case_id]

        if df.empty:
            return {
                "P_EnvFail": 1 / 3,
                "P_AgentRetryFail": 1 / 3,
                "P_AgentReasoningFail": 1 / 3,
            }

        # 使用 EM 模型预测
        post = self.em.predict_proba(df)

        # 聚合为 case 级别
        # 使用阻塞概率公式
        # AgentFail = Retry + Reasoning
        q_agent = np.clip(post[:, 1] + post[:, 2], 0.0, 1.0)
        P_case_AgentFail = 1.0 - float(np.prod((1.0 - q_agent) ** self.alpha))

        # 分解 AgentFail 为 Retry 和 Reasoning
        avg_retry = float(post[:, 1].mean())
        avg_reasoning = float(post[:, 2].mean())
        total_agent = avg_retry + avg_reasoning
        if total_agent > 0:
            P_AgentRetryFail = P_case_AgentFail * (avg_retry / total_agent)
            P_AgentReasoningFail = P_case_AgentFail * (avg_reasoning / total_agent)
        else:
            P_AgentRetryFail = P_case_AgentFail / 2
            P_AgentReasoningFail = P_case_AgentFail / 2

        P_EnvFail = 1.0 - P_case_AgentFail

        return {
            "P_EnvFail": P_EnvFail,
            "P_AgentRetryFail": P_AgentRetryFail,
            "P_AgentReasoningFail": P_AgentReasoningFail,
            "P_AgentFail": P_case_AgentFail,  # 方便使用
        }

    def correct_judgment(
        self,
        agent_original: int,
        case_id: Optional[str] = None,
        tau_agentfail: Optional[float] = None,
        tau_envfail: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        纠正 Agent 的判断

        Args:
            agent_original: Agent 原始判断（0=FAIL, 1=PASS）
            case_id: 测试用例 ID
            tau_agentfail: AgentFail 阈值，为 None 时使用默认值
            tau_envfail: EnvFail 阈值，为 None 时使用默认值

        Returns:
            包含纠偏结果的字典
        """
        if tau_agentfail is None:
            tau_agentfail = self.tau_agentfail
        if tau_envfail is None:
            tau_envfail = self.tau_envfail

        # 获取预测结果
        prediction = self.predict(case_id)
        P_AgentFail = prediction["P_AgentFail"]
        P_EnvFail = prediction["P_EnvFail"]

        action = "keep"
        corrected = agent_original

        if agent_original == 0:  # Agent 判断 FAIL
            if P_AgentFail >= tau_agentfail:
                corrected = 1  # 认定是 AgentFail，翻转为 PASS
                action = "flip_to_AgentFail"
            elif P_EnvFail >= tau_envfail:
                corrected = 0  # 确认是环境问题，维持 FAIL
                action = "keep_EnvFail"
        elif agent_original == 1:  # Agent 判断 PASS
            if P_EnvFail >= tau_envfail:
                corrected = 0  # 环境问题导致误判，翻转为 FAIL
                action = "flip_to_EnvFail"

        return {
            "agent_original": agent_original,
            "corrected_label": corrected,
            "action": action,
            "P_EnvFail": P_EnvFail,
            "P_AgentFail": P_AgentFail,
            "P_AgentRetryFail": prediction["P_AgentRetryFail"],
            "P_AgentReasoningFail": prediction["P_AgentReasoningFail"],
        }

    def correct_with_dataframe(
        self,
        df: pd.DataFrame,
        col_case: str = "test_case_id",
        col_agent: str = "agent_testcase_score_x",
    ) -> pd.DataFrame:
        """
        使用 DataFrame 进行批量纠偏

        Args:
            df: 证据 DataFrame
            col_case: case ID 列名
            col_agent: agent 评分列名

        Returns:
            纠偏结果 DataFrame
        """
        return correct_agent_judgment(
            df=df,
            em=self.em,
            tau_agentfail=self.tau_agentfail,
            tau_envfail=self.tau_envfail,
            alpha=self.alpha,
            col_case=col_case,
            col_agent=col_agent,
        )

    def clear_evidences(self):
        """清空当前收集的证据"""
        self._evidence_rows = []

    def reset(self):
        """重置管理器状态（清空证据并重置学习率）"""
        self.clear_evidences()
        self.em.reset_learning_rate()

    def get_model_params(self) -> Dict[str, Any]:
        """获取当前模型参数"""
        return self.em.get_params()

    def get_online_stats(self) -> Dict[str, Any]:
        """获取在线学习统计信息"""
        stats = self.em.get_online_stats()
        stats["evidence_count"] = len(self._evidence_rows)
        return stats


def create_em_manager(
    params_path: Optional[str] = None,
    enable_online_learning: bool = True,
) -> EMManager:
    """
    创建 EM 管理器的便捷函数

    Args:
        params_path: 预训练参数文件路径
        enable_online_learning: 是否启用在线学习

    Returns:
        EMManager 实例
    """
    return EMManager(
        params_path=params_path,
        enable_online_learning=enable_online_learning,
    )
