"""
EM 模型管理器模块

提供 EMManager 类，整合证据收集、模型预测和纠偏功能。
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from appeval.em.model import SimpleEM4EvidenceH_Refine
from appeval.em.analysis import correct_agent_judgment


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
            os.path.join(os.path.dirname(__file__),
                         "..", "data", "em_params.json"),
            os.path.join(os.path.dirname(__file__),
                         "../../appeval/data/em_params.json"),
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
        from appeval.em.data_process import convert_evidences_to_em_format

        evidences = evidence_collector.get_evidences_for_em()
        project_name = evidence_collector.project_name

        df = convert_evidences_to_em_format(
            evidences=evidences,
            project_name=project_name,
            code_evidence=code_evidence,
            agent_judge={
                project_name: agent_score} if agent_score is not None else None,
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
            P_AgentReasoningFail = P_case_AgentFail * \
                (avg_reasoning / total_agent)
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
