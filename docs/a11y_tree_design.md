# Accessibility Tree (a11y tree) 改造技术文档

## 1. 背景与问题

AppEvalPilot 在 Linux 无头服务器（Xvfb）上运行 Web 应用自动化测试时，需要获取 Chrome 浏览器的 **UI 元素结构化信息**（accessibility tree），用于辅助 OS Agent 理解页面内容和定位交互元素。

原始项目仅支持 Windows 平台（通过 `pywinauto`），Linux 上存在以下问题：

| 问题 | 表现 |
|------|------|
| 无法获取 a11y tree | `get_screen_xml()` 返回空列表，`No valid window found` |
| AT-SPI 服务缺失 | Xvfb 不是完整桌面，不会自动启动 D-Bus / AT-SPI 服务 |
| Chrome 未注册 AT-SPI | 缺少 `GTK_MODULES=gail:atk-bridge` 环境变量 |
| 服务链重量级 | AT-SPI 模式需要 D-Bus + AT-SPI bus launcher + registryd，每个 worker 都要独立一套 |

## 2. 设计方案

引入**双模式架构**，支持两种 accessibility tree 获取方式，用户可按需选择：

```
┌────────────────────────────────────────────────────┐
│               PCController.get_screen_xml()        │
│                   a11y_mode 参数切换                 │
├────────────────────┬───────────────────────────────┤
│   a11y_mode="cdp"  │      a11y_mode="atspi"        │
│  CDPElementProcessor│    LinuxElementProcessor      │
├────────────────────┼───────────────────────────────┤
│ Chrome DevTools    │ D-Bus → AT-SPI Bus →          │
│ Protocol (WS)     │ AT-SPI Registryd → pyatspi     │
│                    │                                │
│ 依赖: 仅 Chrome    │ 依赖: D-Bus + AT-SPI +        │
│ debugging port     │ GTK_MODULES + pyatspi          │
└────────────────────┴───────────────────────────────┘
```

### 2.1 CDP 模式（轻量，存在坐标漂移问题）

**原理**：直接通过 Chrome 已有的 `--remote-debugging-port` WebSocket 接口，调用 `Accessibility.getFullAXTree()` 获取 Chrome 内部的 accessibility tree。

**优点**：
- 零额外服务依赖，不需要 D-Bus / AT-SPI / GTK_MODULES
- 跨平台一致（Windows/Linux/Mac 行为相同）
- 每个 worker 天然隔离（不同端口）

**局限**：
- 仅覆盖**网页内容**（DOM 中的按钮、输入框、链接等），不含 Chrome UI（工具栏、地址栏）
- ⚠️ **坐标漂移问题**：CDP 返回的是**视口相对坐标**（viewport-relative），而 `pyautogui` 使用**屏幕绝对坐标**，存在 ~149px 的 Y 轴偏移（详见 §8 调试记录）

**通信流程**：
```
Python  ──HTTP GET──>  http://127.0.0.1:{port}/json   → 获取 page target
Python  ──WebSocket──> ws://127.0.0.1:{port}/devtools/page/{id}
        ──> Accessibility.enable
        ──> Accessibility.getFullAXTree  → AX Nodes[]
        ──> DOM.getBoxModel(backendNodeId)  → 元素坐标
```

### 2.2 AT-SPI 模式（完整，覆盖 Chrome 整体 UI）

**原理**：通过 Linux AT-SPI（Assistive Technology Service Provider Interface）框架，使用 `pyatspi` 遍历整个应用的 accessibility tree。

**优点**：
- 覆盖 Chrome **整个窗口**（含工具栏、地址栏、页面内容）
- 可获取窗口状态（ACTIVE、FOCUSED、VISIBLE 等）

**局限**：
- 需要启动 D-Bus session bus + AT-SPI bus launcher + AT-SPI registryd
- 需要设置 `GTK_MODULES=gail:atk-bridge` 让 Chrome 加载 ATK bridge
- 每个 worker 需要独立的服务链
- `pyatspi` 依赖系统库，存在 `libffi` 版本冲突风险（Anaconda 环境）

**服务链**：
```
Xvfb → dbus-launch → at-spi-bus-launcher → at-spi2-registryd
                                                    ↑
Chrome (GTK_MODULES=gail:atk-bridge) ──ATK Bridge──┘
                                                    ↓
                                           pyatspi.Registry.getDesktop(0)
```

## 3. 实现详情

### 3.1 修改文件清单

| 文件 | 变更内容 |
|------|---------|
| `appeval/tools/device_controller.py` | 新增 `CDPElementProcessor` 类；`PCController` 增加 `a11y_mode` / `remote_debugging_port` 参数；`get_screen_xml()` 按模式分发；`_iter_top_level_frames()` 增加 `dialog` role 支持 |
| `appeval/roles/osagent.py` | `OSAgent.__init__()` 增加 `a11y_mode` / `remote_debugging_port` 参数；Linux controller_args 透传 |
| `appeval/roles/eval_runner.py` | `agent_params` 增加 `a11y_mode`；`_init_osagent()` 透传到 OSAgent |
| `appeval/utils/window_utils.py` | Chrome 启动增加 `--remote-allow-origins=*`（CDP WebSocket 必需） |
| `run_experiments/run_test.py` | 新增 `_start_dbus_and_atspi()` / `_stop_atspi()` 函数；worker_process 设置 AT-SPI 环境变量；`AppEvalRole` 透传 `a11y_mode` |
| `run_experiments/run_config.yaml` | 新增 `a11y_mode` 配置项 |

### 3.2 CDPElementProcessor 核心实现

```python
class CDPElementProcessor:
    """通过 Chrome DevTools Protocol 获取 accessibility tree"""

    def __init__(self, location_info, max_tokens, remote_debugging_port):
        self.port = remote_debugging_port

    def collect_elements(self) -> List[Dict]:
        # 1. HTTP GET /json → 获取 page target 的 WebSocket URL
        tabs = requests.get(f"http://127.0.0.1:{self.port}/json").json()
        ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]

        # 2. WebSocket 连接
        ws = websocket.create_connection(ws_url)

        # 3. 启用 Accessibility + DOM domain
        _send("Accessibility.enable")
        _send("DOM.enable")

        # 4. 获取完整 AX tree
        nodes = _send("Accessibility.getFullAXTree")["result"]["nodes"]

        # 5. 过滤有意义的节点（跳过 none/generic/StaticText 等）
        # 6. 通过 DOM.getBoxModel(backendNodeId) 获取每个节点的边界框
        # 7. 返回 [{"coordinates": (cx, cy), "text": "text:...; control_type:...; rect: (...)"}]
```

**跳过的无用 role**：`none`, `generic`, `GenericContainer`, `RootWebArea`, `InlineTextBox`, `StaticText`

**坐标获取**：使用 `DOM.getBoxModel` 的 `border` 四边形（8个坐标值），计算 bounding box 和中心点。

### 3.3 AT-SPI 服务链启动（run_test.py worker）

```python
def _start_dbus_and_atspi(display_num, worker_id):
    # 1. dbus-launch --sh-syntax → 获取 DBUS_SESSION_BUS_ADDRESS
    # 2. /usr/libexec/at-spi-bus-launcher &
    # 3. /usr/libexec/at-spi2-registryd &

# worker_process 中设置环境变量：
os.environ["GTK_MODULES"] = "gail:atk-bridge"   # Chrome 加载 ATK bridge
os.environ["GNOME_ACCESSIBILITY"] = "1"          # 全局启用无障碍
os.environ["NO_AT_BRIDGE"] = "0"                 # 确保 bridge 不被禁用
```

### 3.4 Chrome 启动参数（window_utils.py）

```
--force-renderer-accessibility     # 强制启用渲染器的 accessibility
--remote-debugging-port={port}     # CDP 远程调试端口
--remote-allow-origins=*           # 允许 CDP WebSocket 连接（新增）
--no-first-run                     # 跳过首次运行欢迎对话框
--no-default-browser-check         # 跳过默认浏览器检查
--disable-gpu                      # Xvfb 下禁用 GPU（防止黑屏）
--disable-software-rasterizer      # 禁用软件光栅化
--disable-dev-shm-usage            # 避免 /dev/shm 不足
--user-data-dir={path}             # 每个 worker 独立 profile
```

### 3.5 参数传递链路

```
run_config.yaml (a11y_mode)
    ↓
run_test.py: preset.get("a11y_mode", "atspi")
    ↓
AppEvalRole(a11y_mode=..., remote_debugging_port=...)
    ↓
eval_runner.rc.agent_params["a11y_mode"]
    ↓
OSAgent(a11y_mode=..., remote_debugging_port=...)
    ↓
PCController(a11y_mode=..., remote_debugging_port=...)
    ↓
get_screen_xml() → CDPElementProcessor / LinuxElementProcessor
```

## 4. 配置与使用

### 4.1 run_config.yaml 配置

```yaml
# 选择 a11y tree 获取模式
a11y_mode: atspi   # "atspi" (默认推荐，坐标准确) 或 "cdp" (轻量，有坐标漂移)
```

### 4.2 模式选择指南

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| **Web 应用功能测试（默认）** | `atspi` | 坐标准确，覆盖完整窗口，操作可靠 |
| 需要操作 Chrome UI（地址栏等） | `atspi` | AT-SPI 覆盖完整窗口 |
| 资源受限 / 不需要精确坐标 | `cdp` | 无需 D-Bus/AT-SPI 服务，内存占用更低 |
| 需要窗口状态信息 | `atspi` | 可获取 ACTIVE/FOCUSED 等状态 |

> ⚠️ **注意**：由于 CDP 存在坐标漂移问题（详见 §8），默认模式已切换为 `atspi`。CDP 模式仅建议在不依赖 a11y 坐标进行 pyautogui 点击的场景下使用。

### 4.3 依赖安装

**CDP 模式**（最小依赖）：
```bash
pip install websocket-client requests
```

**AT-SPI 模式**（额外依赖）：
```bash
# 系统包
sudo apt install at-spi2-core python3-pyatspi dbus-x11

# Python 包
pip install PyGObject websocket-client requests

# Anaconda 用户可能需要解决 libffi 冲突
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
```

## 5. 调试工具

提供了独立的调试脚本用于验证 a11y tree 获取：

```bash
# 完整测试（自动启动 Xvfb + Chrome + D-Bus + AT-SPI）
./debug_screen_xml_full.sh

# 单独测试（需要先手动启动 Xvfb 和 Chrome）
python debug_screen_xml.py --mode cdp --port 9998
python debug_screen_xml.py --mode atspi
```

**预期输出（CDP 模式）**：
```
CDP: Got 9 AX nodes from Chrome
CDP: Collected 2 elements from Chrome (port 9998)
Elements: 2
  {'coordinates': (959, 26), 'text': 'text:Test; control_type:heading; rect: (8, 8, 1911, 45)'}
  {'coordinates': (39, 76), 'text': 'text:ClickMe; control_type:button; rect: (8, 66, 71, 87)'}
```

**预期输出（AT-SPI 模式）**：
```
Found 1 top-level windows
  [1] Title='Test - Google Chrome' | State=[ACTIVE, SHOWING, VISIBLE]
✓ Using active window: 'Test - Google Chrome'
Collected 25 elements from active window
```

## 6. 已知问题与注意事项

1. **CDP 模式不含 Chrome UI 元素**：地址栏、工具栏按钮等不会出现在 CDP accessibility tree 中。如需操作这些元素，使用 AT-SPI 模式。

2. **AT-SPI 模式的 libffi 冲突**：Anaconda 环境下 `libffi.so.7` 版本可能与系统不一致。代码中已通过 `ctypes.CDLL` 预加载系统 libffi 来解决，也可通过 `LD_PRELOAD` 环境变量解决。

3. **Chrome 首次运行对话框**：`--no-first-run` 可跳过。若未跳过，窗口 role 会是 `dialog` 而非 `frame`，代码已兼容处理（`_iter_top_level_frames` 接受 `dialog` role）。

4. **`--remote-allow-origins=*`**：CDP 模式必需。Chrome 默认拒绝非本地 origin 的 WebSocket 连接，此参数放开限制。

5. **pyautogui 延迟导入**：`device_controller.py` 中 pyautogui 使用 lazy proxy 模式导入，避免在 worker 设置 DISPLAY 前触发连接错误。

## 7. 架构图

```
                    run_config.yaml
                         │
                    run_test.py
                    ┌────┴────┐
                Worker 0   Worker 1  ...  Worker N
                    │         │              │
               ┌────┴────┐   ...            ...
               │  Xvfb   │
               │ :800    │
               │         │
               │ Chrome  │──── port 10000
               │         │         │
               └─────────┘    ┌────┴─────────────┐
                              │  a11y_mode?       │
                              ├──────┬────────────┤
                              │ cdp  │   atspi    │
                              │      │            │
                              │ WS → │ D-Bus →    │
                              │ CDP  │ AT-SPI →   │
                              │ API  │ pyatspi    │
                              └──────┴────────────┘
                                     │
                              get_screen_xml()
                                     │
                              [{coordinates, text}]
                                     │
                              OSAgent perception
```

## 8. CDP 坐标漂移问题——调试记录

### 8.1 问题发现

在 `work_dirs/test2_gemini_screeninfo`（CDP 模式）实验中，观察到大量 **"Mechanics & Focus Failure (W6)"** 错误——agent 按照 a11y tree 给出的坐标点击，但实际点击位置偏离目标元素。

**对比两个实验**：

| 实验 | 模式 | W6 占比 | 典型表现 |
|------|------|---------|---------|
| `test2_gemini_atspi` | AT-SPI | 低 | 点击准确 |
| `test2_gemini_screeninfo` | CDP | **高** | 点击偏移，经常点到上方无关元素 |

### 8.2 根因定位

**选择同一个网页、同一个元素**，分别用 CDP 和 AT-SPI 获取坐标进行对比：

```
目标元素: "Allow" 按钮（Chrome 权限弹窗上的按钮）

CDP  坐标: (394, 174)   ← viewport-relative (CSS pixels from page top-left)
AT-SPI坐标: (394, 323)   ← screen-absolute (pixels from screen top-left)

差值: Δy = 323 - 174 = 149 px
```

**149px 的 Y 轴偏移** 恰好等于 Chrome 浏览器 UI 的高度（标签栏 + 地址栏 + 书签栏 + 安全警告栏）：

```
┌──────────────────────────────────────────┐ ← y=0 (screen top)
│  Chrome 标签栏          (~35px)          │
│  地址栏/导航栏           (~35px)          │
│  --no-sandbox 安全警告栏  (~30px)         │
│  书签栏 (如果有)          (~25px)         │
│  内容与窗口间距           (~24px)         │
├──────────────────────────────────────────┤ ← y≈149 (viewport top)
│                                          │
│           Web Page Content               │
│    CDP 的 (0,0) 从这里开始               │
│                                          │
└──────────────────────────────────────────┘
```

**核心问题**：
- **CDP** 的 `DOM.getBoxModel` 返回的是 **CSS 视口坐标**（从网页内容区域左上角算起）
- **pyautogui** 的 `click(x, y)` 使用的是 **屏幕绝对坐标**（从屏幕左上角算起）
- 两者之间差了 Chrome 浏览器 UI 的高度（~149px），且这个偏移量**不固定**——它取决于：
  - 是否显示 `--no-sandbox` 安全警告栏
  - 是否有书签栏
  - Chrome 窗口是否全屏
  - 屏幕 DPI 缩放比例

### 8.3 影响分析

```
实际点击位置 = CDP 坐标 (x, y)
期望点击位置 = CDP 坐标 (x, y + Chrome_UI_Height)

示例：agent 想点击 y=300 的按钮
  CDP 给出 y=300
  pyautogui.click(x, 300)  → 实际点击了 viewport y=300 对应的屏幕 y=300
  但按钮的屏幕 y 应该是 300 + 149 = 449
  → 点击偏移了 149px，点到了上方不相关的元素
```

**在日志中表现为**：
```
Agent: "I clicked the Allow button at (394, 174)"
验证: "The click landed on the address bar, not the Allow button"
→ MECHANICS_FOCUS_FAILURE (W6)
```

### 8.4 尝试过的修复方案

| 方案 | 实现 | 结果 |
|------|------|------|
| 固定偏移补偿 | `y += 149` | ❌ 偏移量不固定，不同配置/页面不同 |
| 通过 CDP 获取 window bounds | `Browser.getWindowBounds` | ❌ 只返回窗口位置和大小，不含 UI 高度 |
| 通过 CSS `window.innerHeight` 计算 | `Runtime.evaluate("window.outerHeight - window.innerHeight")` | ⚠️ 理论可行，但受 DPI 缩放影响，不够可靠 |
| 使用 `page.getLayoutMetrics` | 可获取 viewport offset | ⚠️ 仅含布局信息，不含窗口装饰 |

### 8.5 最终决策

**放弃修复 CDP 坐标漂移**，改为默认使用 AT-SPI 模式。理由：

1. **AT-SPI 坐标天然是屏幕绝对坐标**，与 `pyautogui` 完全匹配，无需任何转换
2. CDP 的视口偏移量受多种因素影响，无法可靠地自动计算
3. AT-SPI 虽然需要额外的 D-Bus/AT-SPI 服务，但在 `run_test.py` 中已经实现了自动启动和清理
4. AT-SPI 提供**更完整的元素信息**（包含 Chrome UI 元素），对测试更有利

**代码变更**：
- 所有 `a11y_mode` 参数默认值统一改为 `"atspi"`
- `run_config.yaml` 默认配置 `a11y_mode: atspi`
- CDP 模式保留但**不再推荐**，仅作为不需要精确坐标的备选方案

### 8.6 CDP 坐标漂移示例数据

以下是从 `test2_gemini_screeninfo` 实验日志中提取的部分坐标对比（同一页面元素）：

| 元素 | CDP 坐标 (y) | AT-SPI 坐标 (y) | 偏移量 |
|------|-------------|-----------------|--------|
| Block 按钮 | 174 | 323 | +149 |
| Allow 按钮 | 198 | 347 | +149 |
| 页面标题 | 85 | 234 | +149 |
| 正文首段 | 250 | 399 | +149 |
| 底部链接 | 890 | 1039 | +149 |

> 注：同一页面上偏移量一致（149px），但不同页面/配置可能不同。

