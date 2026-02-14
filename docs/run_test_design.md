# run_test.py 跑测脚本设计文档

## 1. 概述

`run_experiments/run_test.py` 是 AppEvalPilot 的批量跑测入口脚本，基于 RealDevBench MGX 数据集，使用多进程并行方式驱动 OS Agent 对 Web 应用进行自动化功能测试，并收集评测结果。

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     main() 主进程                            │
│  1. 加载 run_config.yaml                                     │
│  2. 读取 Excel 数据 (续跑时跳过已完成)                          │
│  3. 构建 task_queue (serial_per_url 分组)                     │
│  4. 启动 N 个 worker 进程                                     │
│  5. 收集结果 (增量保存 + 卡死检测)                              │
│  6. 保存最终 Excel                                           │
├─────────────┬───────────┬──────────┬────────────────────────┤
│  Worker 0   │ Worker 1  │   ...    │     Worker N-1         │
│  Xvfb :800  │ Xvfb :801 │          │     Xvfb :800+N        │
│  Chrome     │ Chrome    │          │     Chrome              │
│  :10000     │ :10001    │          │     :10000+N            │
│  D-Bus      │ D-Bus     │          │     D-Bus               │
│  AT-SPI     │ AT-SPI    │          │     AT-SPI              │
│  GPU 0      │ GPU 1     │          │     GPU N%4             │
│             │           │          │                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 循环: 从 task_queue 取任务                            │    │
│  │   → AppEvalRole.run_api(url, test_cases)             │    │
│  │   → 收集 score/evidence                              │    │
│  │   → 黑屏检测 → 重试                                  │    │
│  │   → result_queue.put(result)                         │    │
│  │   → 每20任务回收Chrome                                │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 3. 配置文件 (`run_config.yaml`)

```yaml
# 运行参数
tasks: 0              # 跑多少个任务，0=全部
workers: 30           # 并行 worker 数
model: remote         # local=本地模型 | remote=远程API
serial_per_url: true  # 同一网站串行、不同网站并行
cuda_devices: [0,1,2,3]  # OCR GPU 列表，worker 轮询分配

# 数据文件
excel_file: "RealDevBench_MGX_20260130.xlsx"
true_label_column: "A 角分数"

# 远程模型配置
remote:
  config_file: config/config_gemini3_flash.yaml
  llm:
    api_type: openai
    model: gemini-3-flash-preview
    base_url: https://newapi.deepwisdom.ai/v1
    api_key: "sk-xxx"
  base_display: 800        # Xvfb 起始 display 号
  base_chrome_port: 10000  # Chrome 起始 debugging port
  log_dir_prefix: test2_gemini
  result_excel: test2_results.xlsx
  score_column: gemini_score
  evidence_column: gemini_evidence
  a11y_mode: atspi         # atspi | cdp
```

### 3.1 默认配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `tasks` | `0` | 0 表示跑全部有效任务 |
| `workers` | `30` | 并行 worker 数，建议不超过 CPU 核数 |
| `model` | `remote` | 使用远程 API 模型（如 Gemini、GPT） |
| `serial_per_url` | `true` | 同一网站的测试点串行执行，避免并发访问冲突 |
| `cuda_devices` | `[0,1,2,3]` | OCR 模型使用的 GPU 列表，worker 按 round-robin 分配 |
| `a11y_mode` | **`atspi`** | Accessibility tree 获取模式（详见下方） |
| `base_display` | `800` | Xvfb 虚拟显示起始编号，worker_id 依次递增 |
| `base_chrome_port` | `10000` | Chrome remote debugging port 起始编号 |

### 3.2 a11y_mode 模式选择

| 模式 | 说明 | 状态 |
|------|------|------|
| **`atspi`**（默认推荐） | 通过 D-Bus → AT-SPI 总线获取完整桌面级无障碍树。坐标为**屏幕绝对坐标**，与 `pyautogui.click(x, y)` 完全匹配。覆盖 Chrome 整个窗口（含工具栏、地址栏、页面内容）。 | ✅ 稳定可用 |
| `cdp` | 通过 Chrome DevTools Protocol 获取浏览器内部无障碍树。无需 D-Bus/AT-SPI 系统服务，更轻量。 | ⚠️ **存在坐标漂移问题** |

#### CDP 坐标漂移问题

CDP 模式下 `DOM.getBoxModel` 返回的是**网页视口相对坐标**（CSS pixels from viewport top-left），而 `pyautogui` 使用的是**屏幕绝对坐标**。两者之间差了 Chrome 浏览器 UI 的高度（标签栏 + 地址栏 + 书签栏 ≈ 149px），导致 agent 的所有点击操作偏移到错误位置。

```
屏幕绝对坐标 y=0 ──────────────────────
  Chrome 标签栏          (~35px)
  地址栏/导航栏           (~35px)
  --no-sandbox 警告栏     (~30px)
  书签栏                  (~25px)
  窗口边距               (~24px)
视口坐标 y=0 ───────────────────────── ← CDP 的 (0,0) 从这里开始
  网页内容区域
```

**实测数据**（同一按钮在两种模式下的坐标）：

| 元素 | CDP 坐标 (y) | AT-SPI 坐标 (y) | 偏移量 |
|------|-------------|-----------------|--------|
| Grid View 按钮 | 432 | 581 | +149px |
| Allow 按钮 | 174 | 323 | +149px |

由于偏移量不固定（取决于 Chrome UI 配置、是否显示警告栏等），CDP 模式下的坐标**不可靠**。因此默认使用 AT-SPI 模式。

> 详细调试记录见 `docs/a11y_tree_design.md` §8。

## 4. 执行流程

### 4.1 主进程 `main()`

```
启动
  │
  ├─ 1. 清理残留进程 (cleanup.sh)
  ├─ 2. 提升 inotify 限制 (sysctl)
  ├─ 3. 加载配置 (load_run_config)
  ├─ 4. 设置 LLM 环境变量 (apply_llm_env)
  ├─ 5. 读取 Excel 数据
  │     ├─ 续跑模式: 读取结果表，跳过已有分数的行
  │     └─ 全新模式: 读取源数据表
  ├─ 6. 构建任务队列
  │     ├─ serial_per_url=true: 按 URL 分组，每组作为一个 job
  │     └─ serial_per_url=false: 每个 task 独立入队
  ├─ 7. 启动 N 个 worker_process
  ├─ 8. 结果收集循环
  │     ├─ 30s 超时轮询 result_queue
  │     ├─ 增量保存 Excel (每 2 分钟)
  │     ├─ worker 存活检测 (全部退出 → 结束)
  │     └─ 卡死检测 (20 分钟无结果 → 强制退出)
  ├─ 9. 终止残留 worker
  └─ 10. 最终保存 Excel + 统计输出
```

### 4.2 Worker 进程 `worker_process()`

每个 worker 是一个独立的 Python 进程，拥有独立的：

| 资源 | 说明 |
|------|------|
| Xvfb | 虚拟显示 `:base_display + worker_id` |
| Chrome | debugging port `base_chrome_port + worker_id` |
| D-Bus | 独立 session bus |
| AT-SPI | 独立 bus launcher + registryd |
| GPU | 轮询分配 `cuda_devices[worker_id % len]` |
| OCR 模型缓存 | `/tmp/modelscope_worker_{id}` |
| Chrome profile | `/tmp/chrome_test2_{prefix}_{id}` |

**Worker 启动序列：**

```
1. 复制 OCR 模型缓存 (从共享目录)
2. 分配 GPU (CUDA_VISIBLE_DEVICES)
3. 设置 LLM 环境变量
4. 启动 Xvfb
5. 设置 QT_QPA_PLATFORM=offscreen (防 cv2 Qt 崩溃)
6. 启动 D-Bus + AT-SPI 服务链
7. 进入任务循环
```

**任务执行循环：**

```
从 task_queue 取一个 item
  │
  ├─ serial_per_url: item 是 list[task]，逐个串行执行
  └─ 否则: item 是单个 task
  
对每个 task:
  │
  ├─ 创建 AppEvalRole 实例
  ├─ 调用 appeval.run_api(url, test_cases)
  │     ├─ 内部: 启动 Chrome → 打开 URL → OS Agent 执行测试步骤
  │     └─ 返回: result dict {"0": {"result": "Pass/Fail", "evidence": "..."}}
  ├─ 解析结果 (兼容 dict 和 str 格式)
  ├─ 黑屏检测
  │     └─ evidence 含 "black screen" → 杀 Chrome → 清 profile → 重试一次
  ├─ result_queue.put(结果)
  └─ 每 20 个任务: pkill 该 worker 的 Chrome (防内存泄漏)
```

## 5. 关键机制

### 5.1 续跑 (Resume)

```python
if args.resume and out_path.exists():
    df = pd.read_excel(out_path)
    valid_df = valid_df[valid_df[score_col].isna()]  # 只跑没分数的
```

- 默认启用 `--resume`
- 读取已有结果 Excel，跳过已有 `score_col` 的行
- 支持多次中断续跑，不丢失已完成的结果

### 5.2 serial_per_url 模式

```python
if serial_per_url:
    by_url = defaultdict(list)
    for t in task_list:
        by_url[t["url"]].append(t)
    for _url, group in by_url.items():
        task_queue.put(group)  # 整组作为一个 job
```

- 同一 URL 的多个测试点在同一 worker 内串行执行
- 不同 URL 之间并行
- 避免多个 worker 同时访问同一网站导致冲突

### 5.3 黑屏检测与重试

```python
if score == 0 and any(k in evidence.lower() for k in black_kws):
    # 杀 Chrome → 清理 profile → 重新创建 AppEvalRole → 重跑该任务
```

- 检测 evidence 中的 "black screen" 等关键词
- 重启 Chrome 并重试一次
- 黑屏率从 ~10% 降至 ~1.5%

### 5.4 Chrome 定期回收

```python
tasks_since_recycle += 1
if tasks_since_recycle >= RECYCLE_EVERY:  # 默认 20
    pkill -f 'user-data-dir={user_data_dir}'
```

- 每 20 个任务杀掉该 worker 的 Chrome 残留进程
- 防止长时间运行导致内存泄漏和黑屏
- 下一个任务会自动启动新 Chrome

### 5.5 卡死检测与增量保存

```python
# 主循环
stall_timeout = 1200  # 20 分钟
save_interval = 120   # 2 分钟

while completed < len(task_list):
    r = result_queue.get(timeout=30)
    # 收到结果 → 更新 last_result_time
    # 每 2 分钟 → 增量保存 Excel
    # 30s 无结果 → 检查 worker 存活
    # 全部 worker 退出 → 结束
    # 20 分钟无结果 → 判定卡死，强制退出
```

### 5.6 LLM 返回格式兜底

```python
res = result["0"]
if isinstance(res, dict):
    result_value = res.get("result", "Fail")
    evidence = res.get("evidence", "")
else:
    # LLM 返回字符串而非 dict
    result_value = str(res)
    evidence = str(res)
```

- LLM 输出格式不稳定时不崩溃
- 从字符串中尝试提取 Pass/Fail

## 6. 环境依赖

### 每个 Worker 启动的进程

| 进程 | 用途 |
|------|------|
| Xvfb | 虚拟 X11 显示 (1920x1080x24) |
| Chrome | 被测 Web 应用的浏览器 |
| dbus-daemon | D-Bus session bus |
| at-spi-bus-launcher | AT-SPI 总线 |
| at-spi2-registryd | AT-SPI 注册守护进程 |

### 系统要求

| 项目 | 要求 |
|------|------|
| inotify.max_user_instances | ≥ 8192 (自动设置) |
| GPU | 每个 worker ~1GB (OCR 模型) |
| 内存 | 每个 worker ~2GB |
| /dev/shm | ≥ 1GB (Chrome 共享内存) |

## 7. 命令行参数

```bash
python run_test.py [OPTIONS]

--config PATH     # 指定 run_config.yaml 路径
--tasks N         # 覆盖任务数 (0=全部)
--workers N       # 覆盖并行 worker 数
--model local|remote  # 覆盖模型选择
--resume          # 续跑模式 (默认启用)
--no-cleanup      # 跳过启动前清理 (默认跳过)
```

## 8. 输出

### Excel 结果表

| 列 | 说明 |
|----|------|
| `case_name` | 应用名称 |
| `prod_id` | 被测 URL |
| `测试点` | 测试用例描述 |
| `A 角分数` | 真实标签 (0/1) |
| `gemini_score` | Agent 评分 (0/1) |
| `gemini_evidence` | Agent 判断依据 |
| `判断一致？` | score == true_label ? 1 : 0 |

### 终端统计

```
=== 统计 ===
测试数量: 794 | 准确率: 77.20%
总耗时: 8766.7s | 平均每任务: 38.5s
Token 消耗: prompt 11577216 + completion 1297602 = 总计 12874818
```

## 9. 已知限制

| 限制 | 说明 | 缓解方案 |
|------|------|---------|
| atoms.dev 登录 | 需要邮箱验证码的注册流程 agent 无法完成 | 提示词注入默认账密 |
| 黑屏 | Chrome 长时间运行后渲染失败 | L2 黑屏重试 + L3 定期回收 |
| OCR GPU 占用 | 每 worker ~1GB | 多卡轮询分配 |
| serial_per_url 尾部 | 大 URL group 串行耗时长 | 20min 卡死超时退出 |
| LLM 格式不稳定 | result 偶尔返回 str | isinstance 兜底 |

