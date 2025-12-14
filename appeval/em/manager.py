"""
EM 模型管理器模块

提供 EMManager 类，整合证据收集、模型预测和纠偏功能。
基于参考实现，使用 case-level 的 posterior 聚合。
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from metagpt.logs import logger as metagpt_logger

from appeval.em.model import SimpleEM4EvidenceH_Refine


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

    # 判断是否需要 retry
    retry_result = em_manager.should_retry(tau_retry=0.4)

    # 进行纠偏
    corrected = em_manager.correct_judgment(agent_original=0)
    ```
    """

    # 默认参数文件路径
    DEFAULT_PARAMS_PATH = "appeval/data/em_params.json"
    DEFAULT_CODE_EVIDENCE_PATH = "appeval/data/webdevjudge_with_code_review_concate.jsonl"

    def __init__(
        self,
        params_path: Optional[str] = None,
        code_evidence_path: Optional[str] = None,
        enable_online_learning: bool = True,
        learning_rate: float = 0.1,
        tau_agentfail: float = 0.7,
        tau_envfail: float = 0.7,
        alpha: float = 0.75,
        # EM model parameters
        w_gui: float = 1.0,
        w_code: float = 1.2,
        w_noresp: float = 0.3,
        agent_weight: float = 0.9,
    ):
        """
        初始化 EM 管理器

        Args:
            params_path: 预训练参数文件路径，为 None 时使用默认参数
            code_evidence_path: code_evidence JSONL 文件路径，为 None 时使用默认路径
            enable_online_learning: 是否启用在线学习
            learning_rate: 在线学习的学习率
            tau_agentfail: AgentFail 判断阈值
            tau_envfail: EnvFail 判断阈值
            alpha: 纠偏时的阻塞概率指数
            w_gui: GUI 证据权重
            w_code: Code 证据权重
            w_noresp: NoResp 证据权重
            agent_weight: Agent 评分权重
        """
        # 使用 metagpt 的 logger，与 osagent 保持一致
        self.logger = metagpt_logger

        # 创建 EM 模型（使用参考实现的参数）
        self.em = SimpleEM4EvidenceH_Refine(
            max_iter=200,
            tol=1e-4,
            seed=42,
            w_gui=w_gui,
            w_code=w_code,
            w_noresp=w_noresp,
            agent_weight=agent_weight,
            a_pi=5.0,
            b_pi=5.0,
            a_c0=3.0,
            b_c0=3.0,
            a_c1=3.0,
            b_c1=3.0,
            a_c2=3.0,
            b_c2=3.0,
            theta_floor=0.05,
            theta_ceil=0.95,
            pi_floor=0.02,
            temp=0.8,
        )

        # 存储 learning_rate 以供在线学习使用
        self.learning_rate = learning_rate

        # 存储证据数据
        self._evidence_rows: List[Dict] = []
        self._current_case_id: str = "default"

        # 纠偏参数
        self.tau_agentfail = tau_agentfail
        self.tau_envfail = tau_envfail
        self.alpha = alpha

        # 在线学习开关
        self.enable_online_learning = enable_online_learning

        # code_evidence 字典: {test_case_id: code_evidence_value}
        self._code_evidence_dict: Dict[str, int] = {}

        # 加载 code_evidence
        if code_evidence_path is not None:
            self.load_code_evidence(code_evidence_path)
        else:
            # 尝试加载默认 code_evidence 文件
            default_code_path = self._find_default_code_evidence_path()
            if default_code_path:
                self.load_code_evidence(default_code_path)
                self.logger.info(f"已加载默认 code_evidence: {default_code_path}")

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

    def _find_default_code_evidence_path(self) -> Optional[str]:
        """查找默认 code_evidence 文件路径"""
        possible_paths = [
            self.DEFAULT_CODE_EVIDENCE_PATH,
            os.path.join(os.path.dirname(__file__),
                         "..", "data", "webdevjudge_with_code_review_concate.jsonl"),
            os.path.join(os.path.dirname(__file__),
                         "../../appeval/data/webdevjudge_with_code_review_concate.jsonl"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def load_code_evidence(self, jsonl_path: str):
        """
        从 JSONL 文件加载 code_evidence 数据

        Args:
            jsonl_path: JSONL 文件路径
        """
        if not os.path.exists(jsonl_path):
            self.logger.warning(f"code_evidence 文件不存在: {jsonl_path}")
            return

        count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    web_id = data.get("web_id", "default")
                    task_id = data.get("task_id", 0)
                    code_review = data.get("code_review", {})
                    is_implemented = code_review.get("is_implemented", False)

                    # 生成 test_case_id，格式: {web_id}_{task_id:02d}
                    test_case_id = f"{web_id}_{task_id:02d}"
                    self._code_evidence_dict[test_case_id] = 1 if is_implemented else 0
                    count += 1
                except json.JSONDecodeError:
                    continue

        self.logger.info(f"已加载 {count} 条 code_evidence 数据")

    def _parse_test_case_id(self, case_id: str) -> Optional[str]:
        """
        从复杂的 case_id 中解析出标准的 test_case_id 格式

        支持的输入格式：
        - 标准格式: "web_21_01" -> "web_21_01"
        - 复杂路径格式: "lqy/xxx/web_21_1_0_174806" -> "web_21_01"

        Args:
            case_id: 原始 case_id

        Returns:
            标准化的 test_case_id（如 "web_21_01"），解析失败返回 None
        """
        import re

        # 如果直接匹配成功，返回原始值
        if case_id in self._code_evidence_dict:
            return case_id

        # 获取最后一部分（处理路径格式）
        last_part = case_id.split("/")[-1]

        # 尝试匹配标准格式 web_X_YY（两位数 task_id）
        standard_match = re.match(r'^(web_\d+)_(\d{2})$', last_part)
        if standard_match:
            standard_id = f"{standard_match.group(1)}_{standard_match.group(2)}"
            if standard_id in self._code_evidence_dict:
                return standard_id

        # 尝试从复杂格式中提取: web_21_1_0_174806 -> web_21, task_id=1
        # 模式: web_{web_num}_{task_id}_{其他部分}
        complex_match = re.match(r'^(web_\d+)_(\d+)_', last_part)
        if complex_match:
            web_id = complex_match.group(1)
            task_id = int(complex_match.group(2))
            parsed_id = f"{web_id}_{task_id:02d}"
            if parsed_id in self._code_evidence_dict:
                self.logger.debug(
                    f"Parsed case_id '{case_id}' -> '{parsed_id}'")
                return parsed_id

        # 尝试更宽松的匹配: 只匹配 web_X_Y 模式
        loose_match = re.match(r'^(web_\d+)_(\d+)', last_part)
        if loose_match:
            web_id = loose_match.group(1)
            task_id = int(loose_match.group(2))
            parsed_id = f"{web_id}_{task_id:02d}"
            if parsed_id in self._code_evidence_dict:
                self.logger.debug(
                    f"Parsed case_id (loose) '{case_id}' -> '{parsed_id}'")
                return parsed_id

        self.logger.debug(f"Failed to parse case_id: '{case_id}'")
        return None

    def get_code_evidence(self, test_case_id: str) -> Optional[int]:
        """
        根据 test_case_id 获取 code_evidence

        支持从复杂的 case_id 格式中自动解析出标准 test_case_id。

        Args:
            test_case_id: 测试用例 ID
                - 标准格式: "web_0_01"
                - 复杂格式: "lqy/xxx/web_21_1_0_174806" (会自动解析为 "web_21_01")

        Returns:
            code_evidence 值（0 或 1），如果找不到返回 None
        """
        # 直接查找
        result = self._code_evidence_dict.get(test_case_id)
        if result is not None:
            return result

        # 尝试解析复杂格式
        parsed_id = self._parse_test_case_id(test_case_id)
        if parsed_id:
            result = self._code_evidence_dict.get(parsed_id)
            if result is not None:
                return result

        # 未找到 code_evidence，报错
        self.logger.warning(
            f"Failed to get code_evidence for test_case_id='{test_case_id}'. "
            f"Parsed id: '{parsed_id}'. "
            f"Available keys sample: {list(self._code_evidence_dict.keys())[:5]}. "
            f"Total loaded: {len(self._code_evidence_dict)} entries."
        )
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

        # 处理代码证据 - 如果未传入，尝试从预加载的字典中查找
        if code_evidence is None:
            # 尝试根据 test_case_id 查找 code_evidence
            code_evidence = self.get_code_evidence(test_case_id)
            if code_evidence is None:
                M_code = 1
                E_code = 0
            else:
                M_code = 0
                E_code = code_evidence
                self.logger.debug(
                    f"自动查找到 code_evidence: {test_case_id} -> {code_evidence}")
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

        # 如果启用在线学习，立即更新模型 TODO
        # if self.enable_online_learning:
        #     self.em.online_update(
        #         gui_evidence=E_gui if M_gui == 0 else None,
        #         code_evidence=E_code if M_code == 0 else None,
        #         agent_score=agent_score,
        #         agent_noresp=E_noresp if M_noresp == 0 else None,
        #         test_case_id=test_case_id,
        #         weight=weight,
        #     )

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

    def _aggregate_case_posteriors(self, df: pd.DataFrame, case_col: str = "test_case_id") -> Dict[str, float]:
        """
        使用 em 的生成参数，对 case 做一次完整贝叶斯：
        P(delta | 所有 step 证据 + agent_testcase_score)

        参考 run_rootcause.py 中的 aggregate_case_posteriors 实现

        Args:
            df: 证据 DataFrame
            case_col: case ID 列名

        Returns:
            包含 case-level 概率的字典

        Raises:
            ValueError: 如果传入的 DataFrame 为空
        """
        eps = 1e-9
        D = 3  # 三类根因

        if df.empty:
            raise ValueError("传入的证据 DataFrame 为空")

        # 对当前 case 的所有 step 聚合
        log_like = np.zeros(D)

        for _, r in df.iterrows():
            for d in range(D):
                p_gui = self.em.theta[d, 0]
                p_code = self.em.theta[d, 1]
                p_no = self.em.theta[d, 2]

                # gui 通道
                if ("M_gui" not in df.columns) or (r.get("M_gui", 0) == 0):
                    e = float(r.get("E1_gui", 0))
                    log_like[d] += np.log((p_gui if e ==
                                          1.0 else 1 - p_gui) + eps)

                # code 通道
                if ("M_code" not in df.columns) or (r.get("M_code", 0) == 0):
                    e = float(r.get("E2_code", 0))
                    log_like[d] += np.log((p_code if e ==
                                          1.0 else 1 - p_code) + eps)

                # noresp 通道
                if ("M_noresp" not in df.columns) or (r.get("M_noresp", 0) == 0):
                    e = float(r.get("E4_noresp", 0))
                    log_like[d] += np.log((p_no if e ==
                                          1.0 else 1 - p_no) + eps)

        # agent_testcase_score 作为 C_case 通道
        C_case = None
        if "agent_testcase_score_x" in df.columns:
            vals = df["agent_testcase_score_x"].dropna().values
            if len(vals) > 0:
                C_case = 1.0 if vals[-1] >= 0.5 else 0.0
        elif "agent_testcase_score" in df.columns:
            vals = df["agent_testcase_score"].dropna().values
            if len(vals) > 0:
                C_case = 1.0 if vals[-1] >= 0.5 else 0.0

        if C_case is not None:
            for d in range(D):
                psi_d = float(self.em.psi[d])
                if C_case == 1.0:
                    log_like[d] += np.log(psi_d + eps)
                else:
                    log_like[d] += np.log(1 - psi_d + eps)

        # prior + 归一化 → posterior
        log_post = np.log(self.em.p_delta + eps) + log_like
        m = log_post.max()
        post = np.exp(log_post - m)
        post = post / post.sum()

        P_env = float(post[0])
        P_retry = float(post[1])
        P_reasoning = float(post[2])
        P_agent = P_retry + P_reasoning

        return {
            "P_case_EnvFail": P_env,
            "P_case_AgentRetryFail": P_retry,
            "P_case_AgentReasoningFail": P_reasoning,
            "P_case_AgentFail": P_agent,
        }

    def predict(self, case_id: str) -> Dict[str, float]:
        """
        预测当前证据的根因概率

        直接调用 EM 模型的 predict_proba 方法

        Args:
            case_id: 测试用例 ID

        Returns:
            包含三类根因概率的字典

        Raises:
            ValueError: 如果没有证据数据或指定的 case_id 不存在
        """

        df = self.get_evidence_dataframe()

        if df.empty:
            raise ValueError(
                "没有证据数据，请先调用 add_evidence 添加证据"
            )

        # 使用指定 case_id 的证据
        if "test_case_id" in df.columns:
            df = df[df["test_case_id"] == case_id]
            if df.empty:
                raise ValueError(
                    f"找不到 case_id={case_id} 的证据数据"
                )
        self.logger.info(f"EM predict_proba for case_id={case_id}")
        # 输出 evidence 和 mask 信息
        evidence_cols = ['E1_gui', 'E2_code', 'E4_noresp']
        mask_cols = ['M_gui', 'M_code', 'M_noresp']
        for idx, row in df.iterrows():
            e_gui = row.get('E1_gui', 'N/A')
            e_code = row.get('E2_code', 'N/A')
            e_noresp = row.get('E4_noresp', 'N/A')
            m_gui = row.get('M_gui', 'N/A')
            m_code = row.get('M_code', 'N/A')
            m_noresp = row.get('M_noresp', 'N/A')
            self.logger.info(
                f"  Step {idx}: Evidence(gui={e_gui}, code={e_code}, noresp={e_noresp}) "
                f"Mask(gui={m_gui}, code={m_code}, noresp={m_noresp})"
            )
        self.logger.info(f"EM model p_delta: {self.em.p_delta}")
        # 直接调用 EM 模型的 predict_proba
        post = self.em.predict_proba(df)
        self.logger.info(f"EM predict_proba result: {post}")

        # 聚合 step-level 到 case-level（取平均）
        P_EnvFail = float(post[:, 0].mean())
        P_AgentRetryFail = float(post[:, 1].mean())
        P_AgentReasoningFail = float(post[:, 2].mean())
        P_AgentFail = P_AgentRetryFail + P_AgentReasoningFail

        return {
            "P_EnvFail": P_EnvFail,
            "P_AgentRetryFail": P_AgentRetryFail,
            "P_AgentReasoningFail": P_AgentReasoningFail,
            "P_AgentFail": P_AgentFail,
        }

    def should_retry(
        self,
        case_id: str,
        tau_retry: float = 0.4,
    ) -> Dict[str, Any]:
        """
        判断是否需要 retry（基于 AgentRetryFail 概率）

        Args:
            case_id: 测试用例 ID（必需）
            tau_retry: retry 判断阈值，当 P_AgentRetryFail 超过此值时返回 True

        Returns:
            包含 retry 判断结果的字典:
            - should_retry: bool，是否需要 retry
            - P_AgentRetryFail: float，AgentRetryFail 概率
            - P_AgentReasoningFail: float，AgentReasoningFail 概率
            - P_EnvFail: float，EnvFail 概率
            - reason: str，判断原因
        """
        prediction = self.predict(case_id)

        P_AgentRetryFail = prediction["P_AgentRetryFail"]
        P_AgentReasoningFail = prediction["P_AgentReasoningFail"]
        P_EnvFail = prediction["P_EnvFail"]

        should_retry = P_AgentRetryFail >= tau_retry

        if should_retry:
            reason = f"AgentRetryFail probability ({P_AgentRetryFail:.3f}) >= threshold ({tau_retry})"
        else:
            if P_EnvFail > P_AgentRetryFail and P_EnvFail > P_AgentReasoningFail:
                reason = f"EnvFail is the most likely cause ({P_EnvFail:.3f})"
            elif P_AgentReasoningFail > P_AgentRetryFail:
                reason = f"AgentReasoningFail is more likely ({P_AgentReasoningFail:.3f}) than retry ({P_AgentRetryFail:.3f})"
            else:
                reason = f"AgentRetryFail probability ({P_AgentRetryFail:.3f}) < threshold ({tau_retry})"

        return {
            "should_retry": should_retry,
            "P_AgentRetryFail": P_AgentRetryFail,
            "P_AgentReasoningFail": P_AgentReasoningFail,
            "P_EnvFail": P_EnvFail,
            "P_AgentFail": prediction["P_AgentFail"],
            "reason": reason,
        }

    def correct_judgment(
        self,
        agent_original: int,
        case_id: str,
        tau_agentfail: Optional[float] = None,
        tau_envfail: Optional[float] = None,
        margin: float = 0.0,
    ) -> Dict[str, Any]:
        """
        纠正 Agent 的判断

        参考 run_rootcause.py 中的 correct_cases_with_post 实现

        Args:
            agent_original: Agent 原始判断（0=FAIL, 1=PASS）
            case_id: 测试用例 ID（必需）
            tau_agentfail: AgentFail 阈值，为 None 时使用默认值
            tau_envfail: EnvFail 阈值，为 None 时使用默认值
            margin: 翻转的边际阈值

        Returns:
            包含纠偏结果的字典
        """
        if tau_agentfail is None:
            tau_agentfail = self.tau_agentfail
        if tau_envfail is None:
            tau_envfail = self.tau_envfail

        # 获取预测结果（使用 case-level 聚合）
        prediction = self.predict(case_id)
        P_AgentFail = prediction["P_AgentFail"]
        P_AgentRetryFail = prediction["P_AgentRetryFail"]
        P_AgentReasoningFail = prediction["P_AgentReasoningFail"]
        P_EnvFail = prediction["P_EnvFail"]

        # 确定失败类型
        agent_fail_type = None
        if P_AgentFail > P_EnvFail + margin:
            if P_AgentRetryFail > P_AgentReasoningFail:
                agent_fail_type = "AgentRetryFail"
            else:
                agent_fail_type = "AgentReasoningFail"

        action = "keep_AgentJudge"
        corrected = agent_original

        if agent_original == 0:  # Agent 判断 FAIL
            # 区分 EnvFail vs AgentFail
            if P_AgentFail > P_EnvFail + margin:
                corrected = 1  # 认定是 AgentFail，翻转为 PASS
                if agent_fail_type == "AgentRetryFail":
                    action = "flip_to_AgentRetryFail"
                else:
                    action = "flip_to_AgentReasoningFail"
            else:
                corrected = 0  # 保持 FAIL
                action = "keep_EnvFail_or_AgentFail"
        elif agent_original == 1:  # Agent 判断 PASS
            # 只在强 EnvFail 证据下翻转
            if P_EnvFail > P_AgentFail + margin:
                corrected = 0  # 环境问题导致误判，翻转为 FAIL
                action = "flip_to_EnvFail"
            else:
                corrected = 1  # 保持 PASS
                action = "keep_AgentJudge"

        return {
            "agent_original": agent_original,
            "corrected_label": corrected,
            "action": action,
            "P_EnvFail": P_EnvFail,
            "P_AgentFail": P_AgentFail,
            "P_AgentRetryFail": P_AgentRetryFail,
            "P_AgentReasoningFail": P_AgentReasoningFail,
            "agent_fail_type": agent_fail_type,
        }

    def correct_with_dataframe(
        self,
        df: pd.DataFrame,
        col_case: str = "test_case_id",
        col_agent: str = "agent_testcase_score_x",
        margin: float = 0.0,
    ) -> pd.DataFrame:
        """
        使用 DataFrame 进行批量纠偏

        参考 run_rootcause.py 中的 correct_cases_with_post 实现

        Args:
            df: 证据 DataFrame
            col_case: case ID 列名
            col_agent: agent 评分列名
            margin: 翻转的边际阈值

        Returns:
            纠偏结果 DataFrame
        """
        rows = []

        for cid, g in df.groupby(col_case):
            g = g.sort_index()

            # 计算 case-level posterior
            probs = self._aggregate_case_posteriors(g, col_case)
            P_env = probs["P_case_EnvFail"]
            P_retry = probs["P_case_AgentRetryFail"]
            P_reasoning = probs["P_case_AgentReasoningFail"]
            P_agent = probs["P_case_AgentFail"]

            # 获取 agent 原始判定
            C_case = None
            if col_agent in g.columns:
                vals = g[col_agent].dropna().values
                if len(vals) > 0:
                    C_case = int(vals[-1])

            # 确定失败类型
            agent_fail_type = None
            if P_agent > P_env + margin:
                if P_retry > P_reasoning:
                    agent_fail_type = "AgentRetryFail"
                else:
                    agent_fail_type = "AgentReasoningFail"

            if C_case is None or np.isnan(C_case):
                # 没原判定：直接用 argmax
                corrected = 1 if P_agent >= P_env else 0
                action = "from_model"
            elif C_case == 0:
                # agent 说 FAIL：区分 EnvFail vs AgentFail
                if P_agent > P_env + margin:
                    corrected = 1
                    if agent_fail_type == "AgentRetryFail":
                        action = "flip_to_AgentRetryFail"
                    else:
                        action = "flip_to_AgentReasoningFail"
                else:
                    corrected = 0
                    action = "keep_EnvFail_or_AgentFail"
            else:  # C_case == 1, agent 说 PASS
                # 只在强 EnvFail 证据下翻转
                if P_env > P_agent + margin:
                    corrected = 0
                    action = "flip_to_EnvFail"
                else:
                    corrected = 1
                    action = "keep_AgentJudge"

            rows.append(
                dict(
                    case_id=cid,
                    agent_original=C_case,
                    P_case_EnvFail=P_env,
                    P_case_AgentRetryFail=P_retry,
                    P_case_AgentReasoningFail=P_reasoning,
                    P_case_AgentFail=P_agent,
                    corrected_label=corrected,
                    action=action,
                    agent_fail_type=agent_fail_type,
                )
            )

        return pd.DataFrame(rows).sort_values("case_id") if rows else pd.DataFrame()

    def clear_evidences(self):
        """清空当前收集的证据"""
        self._evidence_rows = []

    def reset(self):
        """重置管理器状态（清空证据）"""
        self.clear_evidences()

    def get_model_params(self) -> Dict[str, Any]:
        """获取当前模型参数"""
        return self.em.get_params()

    def get_online_stats(self) -> Dict[str, Any]:
        """获取在线学习统计信息"""
        stats = {
            "learning_rate": self.learning_rate,
            "enable_online_learning": self.enable_online_learning,
        }
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
