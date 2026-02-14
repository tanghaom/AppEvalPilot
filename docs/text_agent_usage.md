# TextAgent 使用指南

> 纯文本 Web 测试 Agent，基于无障碍树 (Accessibility Tree) 操作网页，不依赖截图/VLM。

## 一、架构概述

```
OSAgent (现有, VLM + 截图)
    │  继承
    ▼
TextAgent (新, 纯文本 + a11y tree)
    │  被引用
    ▼
AppEvalRole (agent_class="text_agent")
    │  被调用
    ▼
run_test.py (model: text)  /  main_text.py (单次测试)
```

### 与 OSAgent 对比

| 维度 | OSAgent (remote) | TextAgent (text) |
|------|-----------------|-----------------|
| 感知方式 | 截图 + OCR + a11y tree | 仅 a11y tree |
| LLM 类型 | VLM（需要图像理解） | 纯文本 LLM |
| Token 消耗 | ~14万/任务 | ~1万/任务 (10x↓) |
| 平均耗时 | ~275s/任务 | ~68s/任务 (4x↓) |
| GPU 需求 | 需要（OCR 模型） | 不需要 |
| 适用场景 | 通用 GUI 测试 | 功能/交互测试 |
| 不适用 | — | 纯视觉测试（颜色、动画、Canvas、图片内容） |

## 二、文件清单

| 文件 | 说明 |
|------|------|
| `appeval/roles/text_agent.py` | TextAgent 类，继承 OSAgent |
| `appeval/prompts/text_agent.py` | 纯文本 prompt 模板 |
| `appeval/roles/eval_runner.py` | 已支持 `agent_class="text_agent"` |
| `run_experiments/run_config.yaml` | 已添加 `text` 配置段 |
| `run_experiments/run_test.py` | 已支持 `--model text` |
| `run_experiments/check_results.py` | 结果检查/对比脚本 |
| `main_text.py` | 单次测试入口 |
| `test_run_text.sh` | 单次测试快速启动脚本 |

## 三、批量跑测（run_test.py）

### 3.1 配置 run_config.yaml

```yaml
# 切换到 TextAgent 模式，只需改这一行
model: text     # local | remote | text

# text 配置段
text:
  config_file: config/config_gemini3_flash.yaml
  agent_class: text_agent
  a11y_mode: cdp               # cdp（推荐）或 atspi
  llm:
    api_type: openai
    model: gemini-3-flash-preview
    base_url: https://newapi.deepwisdom.ai/v1
    api_key: sk-你的Key
    stream: "false"
  base_display: 1200
  base_chrome_port: 11000
  max_iters: 15
  use_ocr: false
  debug_screenshots: true      # 截图存盘供人工审查，不送 LLM
  log_dir_prefix: test2_text_agent
  result_excel: test2_text_agent_results.xlsx
  score_column: text_agent_score
  evidence_column: text_agent_evidence
```

### 3.2 启动跑测

```bash
cd run_experiments

# 方式 1: 使用 yaml 配置
python run_test.py

# 方式 2: 命令行覆盖
python run_test.py --model text

# 方式 3: 指定任务数和并行数
python run_test.py --model text --tasks 100 --workers 30

# 方式 4: 续跑（默认行为，跳过已有分数的任务）
python run_test.py  # --resume 默认启用
```

### 3.3 a11y_mode 选择

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `cdp` | Chrome DevTools Protocol，轻量 | 纯网页测试（推荐） |
| `atspi` | AT-SPI，桌面级无障碍树 | 涉及系统文件对话框（如文件上传） |

**区别**：`cdp` 只能看到 Chrome 内的网页元素；`atspi` 能看到整个桌面（包括系统弹窗）。

## 四、单次测试（main_text.py）

### 4.1 快速启动

```bash
bash test_run_text.sh
```

### 4.2 直接运行

```bash
export DISPLAY=:99
export QT_QPA_PLATFORM=offscreen
# 确保 Xvfb 已启动
python main_text.py --platform Linux --max_iters 10
```

### 4.3 代码调用

```python
from appeval.roles.eval_runner import AppEvalRole

appeval = AppEvalRole(
    json_file="data/test.json",
    agent_class="text_agent",
    a11y_mode="cdp",
    remote_debugging_port=9333,
    user_data_dir="/tmp/chrome_text_agent",
    extend_xml_infos=True,
    max_iters=10,
    debug_screenshots=True,
)

result, executability = await appeval.run_api(
    task_name="MyTest",
    test_cases={"0": {"case_desc": "验证登录功能", "result": "", "evidence": ""}},
    start_func="https://example.com",
    log_dir="work_dirs_text",
)
```

## 五、结果检查（check_results.py）

```bash
cd run_experiments

# 查看 TextAgent 结果（默认）
python check_results.py

# 查看指定结果文件
python check_results.py --file test2_sf_api_atspi_test_gemini_3_flash_preview_c46o.xlsx

# 查看 20 个错误案例详情
python check_results.py --errors 20

# 对比两个模型的结果
python check_results.py --compare \
    test2_text_agent_results.xlsx \
    test2_sf_api_atspi_test_gemini_3_flash_preview_c46o.xlsx

# 实时监控（跑测期间每 30s 刷新）
python check_results.py --watch 30

# 自定义列名
python check_results.py --file xxx.xlsx --score my_score --evidence my_evidence
```

### 输出示例

```
📊 结果文件: test2_text_agent_results.xlsx
总任务: 816 | 已完成: 185 | 有真实标签: 185

🎯 准确率: 52.73%

📋 混淆矩阵:
                   真实 Pass    真实 Fail
  预测 Pass:         30  (TP)      5  (FP)
  预测 Fail:         47  (FN)     28  (TN)

  Pass 召回率: 39.0%
  Pass 精确率: 85.7%
  Fail 召回率: 84.8%
```

## 六、已知局限

### 6.1 不可感知的测试类型（~17%）

| 类型 | 数量 | 原因 |
|------|------|------|
| 颜色/样式 | ~18 | a11y tree 无颜色信息 |
| 图片内容 | ~21 | 能知道 img 元素存在，但看不到内容 |
| 布局/响应式 | ~12 | 无法判断视觉排列 |
| 动画/过渡 | ~21 | 完全不可感知 |
| 3D/Canvas | ~17 | Canvas 内容在 tree 里是黑盒 |
| 文件上传（cdp 模式） | ~18 | cdp 看不到系统文件对话框 |

### 6.2 优化方向

1. **提升 Pass 召回率**：当前 TextAgent 偏向判 Fail，可通过 prompt 优化引导更积极的判断
2. **增加页面加载等待**：CDP 首次取元素时页面可能未加载完
3. **切 atspi 模式**：覆盖文件上传场景
4. **混合模式**：纯文本 agent 先跑一轮，视觉类测试点再用 VLM agent 补测

