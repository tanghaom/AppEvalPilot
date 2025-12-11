"""
EM 模型核心模块

实现 SimpleEM4EvidenceH_Refine 模型，用于根因分析。
"""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class SimpleEM4EvidenceH_Refine:
    """
    基于 EM 算法的证据融合模型

    用于融合多种证据（GUI证据、代码证据、反思证据等）进行根因分析。
    支持三种根因类型：
    - EnvFail (0): 环境问题
    - AgentRetryFail (1): Agent重试失败
    - AgentReasoningFail (2): Agent推理失败
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        learning_rate: float = 0.1,
        random_state: Optional[int] = None,
    ):
        """
        初始化 EM 模型

        Args:
            n_components: 隐变量类别数（默认3：EnvFail, AgentRetryFail, AgentReasoningFail）
            max_iter: 最大迭代次数
            tol: 收敛阈值
            learning_rate: 在线学习的学习率
            random_state: 随机种子
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.logger = logging.getLogger(
            f"{__name__}.SimpleEM4EvidenceH_Refine")

        # 模型参数
        self._pi = None  # 先验概率 P(H)
        self._theta_gui = None  # P(E_gui | H)
        self._theta_code = None  # P(E_code | H)
        self._theta_reflect = None  # P(E_reflect | H)
        self._theta_noresp = None  # P(E_noresp | H)

        # 在线学习统计
        self._online_samples = 0
        self._initial_learning_rate = learning_rate

        # 初始化标志
        self._is_initialized = False

    def _initialize_params(self):
        """初始化模型参数"""
        np.random.seed(self.random_state)

        # 先验概率 P(H)
        self._pi = np.array([1 / 3, 1 / 3, 1 / 3])

        # 条件概率 P(E | H)
        # 每个证据对每个隐变量的条件概率
        # shape: (n_components, 2) 表示 P(E=0|H), P(E=1|H)

        # GUI 证据：命中与否
        # EnvFail: GUI可能正常也可能异常
        # AgentRetryFail: GUI更可能异常
        # AgentReasoningFail: GUI更可能正常
        self._theta_gui = np.array([
            [0.4, 0.6],  # EnvFail: P(gui=0|H0), P(gui=1|H0)
            [0.7, 0.3],  # AgentRetryFail: P(gui=0|H1), P(gui=1|H1)
            [0.3, 0.7],  # AgentReasoningFail: P(gui=0|H2), P(gui=1|H2)
        ])

        # Code 证据：实现与否
        self._theta_code = np.array([
            [0.8, 0.2],  # EnvFail: 环境问题时代码可能未实现
            [0.4, 0.6],  # AgentRetryFail: 重试失败时代码可能已实现
            [0.3, 0.7],  # AgentReasoningFail: 推理失败时代码更可能已实现
        ])

        # Reflect 证据：反思结果
        self._theta_reflect = np.array([
            [0.5, 0.5],  # EnvFail: 均匀
            [0.6, 0.4],  # AgentRetryFail
            [0.4, 0.6],  # AgentReasoningFail
        ])

        # NoResp 证据：无响应/异常
        self._theta_noresp = np.array([
            [0.6, 0.4],  # EnvFail: 环境问题更可能导致无响应
            [0.5, 0.5],  # AgentRetryFail
            [0.7, 0.3],  # AgentReasoningFail: 推理失败不太可能无响应
        ])

        self._is_initialized = True

    def _e_step(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        E步：计算后验概率 P(H | E)

        Args:
            X: 证据矩阵 (n_samples, n_features)
            M: mask矩阵 (n_samples, n_features), 1表示缺失

        Returns:
            后验概率矩阵 (n_samples, n_components)
        """
        n_samples = X.shape[0]
        posterior = np.zeros((n_samples, self.n_components))

        for i in range(n_samples):
            log_prob = np.log(self._pi + 1e-10)

            # GUI 证据
            if M[i, 0] == 0:  # 非缺失
                gui_val = int(X[i, 0])
                log_prob += np.log(self._theta_gui[:, gui_val] + 1e-10)

            # Code 证据
            if M[i, 1] == 0:
                code_val = int(X[i, 1])
                log_prob += np.log(self._theta_code[:, code_val] + 1e-10)

            # Reflect 证据（可能是连续值，需要离散化）
            if M[i, 2] == 0:
                reflect_val = 1 if X[i, 2] > 0.5 else 0
                log_prob += np.log(self._theta_reflect[:, reflect_val] + 1e-10)

            # NoResp 证据
            if M[i, 3] == 0:
                noresp_val = int(X[i, 3])
                log_prob += np.log(self._theta_noresp[:, noresp_val] + 1e-10)

            # 归一化
            log_prob -= np.max(log_prob)  # 防止数值溢出
            prob = np.exp(log_prob)
            posterior[i] = prob / (prob.sum() + 1e-10)

        return posterior

    def _m_step(self, X: np.ndarray, M: np.ndarray, posterior: np.ndarray):
        """
        M步：更新模型参数

        Args:
            X: 证据矩阵
            M: mask矩阵
            posterior: 后验概率矩阵
        """
        n_samples = X.shape[0]

        # 更新先验
        self._pi = posterior.sum(axis=0) / n_samples
        self._pi = np.clip(self._pi, 0.01, 0.99)
        self._pi /= self._pi.sum()

        # 更新条件概率
        for h in range(self.n_components):
            # GUI
            mask_gui = M[:, 0] == 0
            if mask_gui.sum() > 0:
                for v in [0, 1]:
                    numer = ((X[mask_gui, 0] == v) *
                             posterior[mask_gui, h]).sum()
                    denom = posterior[mask_gui, h].sum()
                    self._theta_gui[h, v] = (numer + 0.1) / (denom + 0.2)

            # Code
            mask_code = M[:, 1] == 0
            if mask_code.sum() > 0:
                for v in [0, 1]:
                    numer = ((X[mask_code, 1] == v) *
                             posterior[mask_code, h]).sum()
                    denom = posterior[mask_code, h].sum()
                    self._theta_code[h, v] = (numer + 0.1) / (denom + 0.2)

            # NoResp
            mask_noresp = M[:, 3] == 0
            if mask_noresp.sum() > 0:
                for v in [0, 1]:
                    numer = ((X[mask_noresp, 3] == v) *
                             posterior[mask_noresp, h]).sum()
                    denom = posterior[mask_noresp, h].sum()
                    self._theta_noresp[h, v] = (numer + 0.1) / (denom + 0.2)

    def fit(self, df: pd.DataFrame, weights: Optional[np.ndarray] = None) -> "SimpleEM4EvidenceH_Refine":
        """
        训练模型

        Args:
            df: 包含证据列的 DataFrame
            weights: 样本权重

        Returns:
            self
        """
        if not self._is_initialized:
            self._initialize_params()

        # 提取特征
        X, M = self._extract_features(df)

        # EM 迭代
        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            # E步
            posterior = self._e_step(X, M)

            # M步
            self._m_step(X, M, posterior)

            # 计算对数似然
            ll = self._log_likelihood(X, M)

            # 检查收敛
            if abs(ll - prev_ll) < self.tol:
                self.logger.info(f"EM 收敛于第 {iteration + 1} 次迭代")
                break

            prev_ll = ll

        return self

    def partial_fit(self, df: pd.DataFrame) -> "SimpleEM4EvidenceH_Refine":
        """
        在线学习：增量更新模型

        Args:
            df: 新的证据数据

        Returns:
            self
        """
        if not self._is_initialized:
            self._initialize_params()

        X, M = self._extract_features(df)

        # E步
        posterior = self._e_step(X, M)

        # 使用学习率进行增量更新
        n_samples = X.shape[0]
        lr = self.learning_rate / (1 + self._online_samples * 0.01)

        # 更新先验
        new_pi = posterior.sum(axis=0) / n_samples
        self._pi = (1 - lr) * self._pi + lr * new_pi
        self._pi /= self._pi.sum()

        self._online_samples += n_samples

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
        单样本在线更新

        Args:
            gui_evidence: GUI 证据
            code_evidence: 代码证据
            agent_score: Agent 评分
            agent_noresp: 无响应证据
            test_case_id: 测试用例 ID
            weight: 样本权重
        """
        if not self._is_initialized:
            self._initialize_params()

        # 构建单样本数据
        data = {
            "test_case_id": [test_case_id],
            "E1_gui": [gui_evidence if gui_evidence is not None else 0],
            "E2_code": [code_evidence if code_evidence is not None else 0],
            "E3_reflect": [agent_score if agent_score is not None else np.nan],
            "E4_noresp": [agent_noresp if agent_noresp is not None else 0],
            "M_gui": [0 if gui_evidence is not None else 1],
            "M_code": [0 if code_evidence is not None else 1],
            "M_reflect": [0 if agent_score is not None else 1],
            "M_noresp": [0 if agent_noresp is not None else 1],
            "weight": [weight],
        }
        df = pd.DataFrame(data)
        self.partial_fit(df)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        预测后验概率

        Args:
            df: 证据 DataFrame

        Returns:
            后验概率矩阵 (n_samples, n_components)
        """
        if not self._is_initialized:
            self._initialize_params()

        X, M = self._extract_features(df)
        return self._e_step(X, M)

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

    def _extract_features(self, df: pd.DataFrame) -> tuple:
        """
        从 DataFrame 提取特征和 mask

        Args:
            df: 证据 DataFrame

        Returns:
            (X, M) 特征矩阵和 mask 矩阵
        """
        n_samples = len(df)

        # 特征矩阵
        X = np.zeros((n_samples, 4))
        M = np.ones((n_samples, 4))  # 默认全部缺失

        # GUI 证据
        if "E1_gui" in df.columns:
            X[:, 0] = df["E1_gui"].fillna(0).values
            if "M_gui" in df.columns:
                M[:, 0] = df["M_gui"].values
            else:
                M[:, 0] = 0

        # Code 证据
        if "E2_code" in df.columns:
            X[:, 1] = df["E2_code"].fillna(0).values
            if "M_code" in df.columns:
                M[:, 1] = df["M_code"].values
            else:
                M[:, 1] = 0

        # Reflect 证据
        if "E3_reflect" in df.columns:
            X[:, 2] = df["E3_reflect"].fillna(0.5).values
            if "M_reflect" in df.columns:
                M[:, 2] = df["M_reflect"].values
            else:
                M[:, 2] = df["E3_reflect"].isna().astype(int).values

        # NoResp 证据
        if "E4_noresp" in df.columns:
            X[:, 3] = df["E4_noresp"].fillna(0).values
            if "M_noresp" in df.columns:
                M[:, 3] = df["M_noresp"].values
            else:
                M[:, 3] = 0

        return X, M

    def _log_likelihood(self, X: np.ndarray, M: np.ndarray) -> float:
        """计算对数似然"""
        n_samples = X.shape[0]
        ll = 0.0

        for i in range(n_samples):
            prob = np.zeros(self.n_components)
            for h in range(self.n_components):
                log_p = np.log(self._pi[h] + 1e-10)

                if M[i, 0] == 0:
                    log_p += np.log(self._theta_gui[h, int(X[i, 0])] + 1e-10)
                if M[i, 1] == 0:
                    log_p += np.log(self._theta_code[h, int(X[i, 1])] + 1e-10)
                if M[i, 2] == 0:
                    reflect_val = 1 if X[i, 2] > 0.5 else 0
                    log_p += np.log(self._theta_reflect[h,
                                    reflect_val] + 1e-10)
                if M[i, 3] == 0:
                    log_p += np.log(self._theta_noresp[h,
                                    int(X[i, 3])] + 1e-10)

                prob[h] = np.exp(log_p)

            ll += np.log(prob.sum() + 1e-10)

        return ll

    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        if not self._is_initialized:
            self._initialize_params()

        return {
            "pi": self._pi.tolist(),
            "theta_gui": self._theta_gui.tolist(),
            "theta_code": self._theta_code.tolist(),
            "theta_reflect": self._theta_reflect.tolist(),
            "theta_noresp": self._theta_noresp.tolist(),
            "online_samples": self._online_samples,
        }

    def load_params(self, params: Dict[str, Any]):
        """加载模型参数"""
        self._pi = np.array(params["pi"])
        self._theta_gui = np.array(params["theta_gui"])
        self._theta_code = np.array(params["theta_code"])
        self._theta_reflect = np.array(params["theta_reflect"])
        self._theta_noresp = np.array(params["theta_noresp"])
        self._online_samples = params.get("online_samples", 0)
        self._is_initialized = True

    def save_params(self, filepath: str):
        """保存模型参数到文件"""
        params = self.get_params()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

    def reset_learning_rate(self):
        """重置学习率"""
        self.learning_rate = self._initial_learning_rate
        self._online_samples = 0

    def get_online_stats(self) -> Dict[str, Any]:
        """获取在线学习统计"""
        return {
            "online_samples": self._online_samples,
            "current_learning_rate": self.learning_rate / (1 + self._online_samples * 0.01),
            "pi": self._pi.tolist() if self._pi is not None else None,
        }
