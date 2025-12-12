"""
EM 模型核心模块

实现 SimpleEM4EvidenceH_Refine 模型，用于根因分析。
基于参考实现，支持三类根因：δ ∈ {0=EnvFail, 1=AgentRetryFail, 2=AgentReasoningFail}
"""

import json
import logging
from typing import Any, Dict, Optional

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
        # Legacy compatibility parameters (ignored, for backward compatibility)
        n_components: int = 3,
        learning_rate: float = 0.1,
        random_state: Optional[int] = None,
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

        # parameters
        # [EnvFail, AgentRetryFail, AgentReasoningFail]
        self.p_delta = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        # gui/code/noresp for 3 classes
        self.theta = np.full((3, 3), 0.5, dtype=float)
        # P(C=1|EnvFail), P(C=1|AgentRetryFail), P(C=1|AgentReasoningFail)
        self.psi = np.array([0.5, 0.5, 0.5], dtype=float)

        # columns (fit 时覆盖)
        self.col_gui = "E1_gui"
        self.col_code = "E2_code"
        self.col_noresp = "E4_noresp"
        self.col_w = "weight"
        self.col_case = "test_case_id"
        self.col_agent = "agent_testcase_score_x"
        self.col_delta = "delta_label"

        # 初始化标志（用于兼容）
        self._is_initialized = False

        # 在线学习统计（用于兼容）
        self._online_samples = 0
        self._initial_learning_rate = learning_rate
        self.learning_rate = learning_rate

        self.logger = logging.getLogger(f"{__name__}.SimpleEM4EvidenceH_Refine")

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
        self._is_initialized = True

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
        weights: Optional[np.ndarray] = None,
    ):  # weights for backward compatibility
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
                    self.psi[2] = float(np.clip((ones_list[2] + self.a_c2 - 1) / (den_list[2] + self.a_c2 + self.b_c2 - 2 + eps), 0.02, 0.98))

            # (4) LL convergence
            avg_ll = float((w * log_den.squeeze()).sum() / (w.sum() + eps))
            if abs(avg_ll - ll_prev) < self.tol:
                self.logger.info(f"EM 收敛于第 {it + 1} 次迭代")
                break
            ll_prev = avg_ll

        return self

    # ---------- inference ----------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """返回 step-level P(EnvFail), P(AgentRetryFail), P(AgentReasoningFail)，使用与 fit 一致的 M_* 掩码"""
        if not self._is_initialized:
            self._init_params(np.zeros((1, 3)))

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

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        预测最可能的根因类别

        Args:
            df: 证据 DataFrame

        Returns:
            预测类别数组
        """
        proba = self.predict_proba(df)
        return np.argmax(proba, axis=1)

    def get_params(self) -> Dict[str, Any]:
        """获取模型参数（参考格式）"""
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
        支持参考格式的参数
        """
        if "P_delta" in params:
            # 参考格式
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
        elif "pi" in params:
            # 旧格式兼容（如果有）
            self.p_delta = np.array(params["pi"][:3])
            # 其他参数使用默认值
            self.logger.warning("使用旧格式参数，部分参数可能不准确")

        self._is_initialized = True

    def save_params(self, filepath: str):
        """保存模型参数到文件"""
        params = self.get_params()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

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

    # ---------- 在线学习接口（用于 osagent.py 兼容）----------

    def partial_fit(self, df: pd.DataFrame) -> "SimpleEM4EvidenceH_Refine":
        """
        在线学习：增量更新模型（简化实现，用于兼容）

        Args:
            df: 新的证据数据

        Returns:
            self
        """
        if not self._is_initialized:
            self._init_params(np.zeros((1, 3)))

        # 简化的在线更新：使用当前数据做一轮 EM 更新
        E, w, M, case_ids, C_raw, delta_sup_raw = self._extract(df)
        N = E.shape[0]

        # 使用学习率进行增量更新
        lr = self.learning_rate / (1 + self._online_samples * 0.01)

        # 计算当前 posterior
        post = self.predict_proba(df)

        # 更新先验
        new_pi = post.mean(axis=0)
        self.p_delta = (1 - lr) * self.p_delta + lr * new_pi
        self.p_delta /= self.p_delta.sum()

        self._online_samples += N

        return self

    def online_update(
        self,
        gui_evidence: Optional[int] = None,
        code_evidence: Optional[int] = None,
        agent_score: Optional[float] = None,
        agent_noresp: Optional[int] = None,
        test_case_id: str = "default",
        weight: float = 1.0,
    ):
        """
        单样本在线更新（用于 osagent.py 兼容）

        Args:
            gui_evidence: GUI 证据
            code_evidence: 代码证据
            agent_score: Agent 评分
            agent_noresp: 无响应证据
            test_case_id: 测试用例 ID
            weight: 样本权重
        """
        if not self._is_initialized:
            self._init_params(np.zeros((1, 3)))

        # 构建单样本数据
        data = {
            "test_case_id": [test_case_id],
            "E1_gui": [gui_evidence if gui_evidence is not None else 0],
            "E2_code": [code_evidence if code_evidence is not None else 0],
            "E4_noresp": [agent_noresp if agent_noresp is not None else 0],
            "M_gui": [0 if gui_evidence is not None else 1],
            "M_code": [0 if code_evidence is not None else 1],
            "M_noresp": [0 if agent_noresp is not None else 1],
            "weight": [weight],
        }
        if agent_score is not None:
            data["agent_testcase_score_x"] = [agent_score]
        df = pd.DataFrame(data)
        self.partial_fit(df)

    def reset_learning_rate(self):
        """重置学习率"""
        self.learning_rate = self._initial_learning_rate
        self._online_samples = 0

    def get_online_stats(self) -> Dict[str, Any]:
        """获取在线学习统计"""
        return {
            "online_samples": self._online_samples,
            "current_learning_rate": self.learning_rate / (1 + self._online_samples * 0.01),
            "pi": self.p_delta.tolist() if self.p_delta is not None else None,
        }
