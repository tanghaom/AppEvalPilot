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
                    self.psi[2] = float(np.clip((ones_list[2] + self.a_c2 - 1) / (den_list[2] + self.a_c2 + self.b_c2 - 2 + eps), 0.02, 0.98))

            # (4) LL convergence
            avg_ll = float((w * log_den.squeeze()).sum() / (w.sum() + eps))
            if abs(avg_ll - ll_prev) < self.tol:
                break
            ll_prev = avg_ll

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
