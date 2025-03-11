# AppEvalPilot

## 介绍

AppEvalPilot是一个软件应用功能完整性评估自动化工具。它专为桌面应用、移动/App应用和Web应用设计。

AppEvalPilot可以帮助您全自动评估任何应用，无需人工干预，节省时间和资源的同时保持高准确度。

在2000多个测试样例的评估中，AppEvalPilot与人类专家的判断高度相关（所有版本的Pearson相关系数为0.9249，平均Spearman相关系数为0.9021）。

### 特性

1. **使用便捷**：一套代码评估桌面应用、移动/App应用、Web应用。
   
2. **稳健又可靠的动态评估**：不同于其他benchmark测评使用的静态评估方式，我们是通过模拟网页测试工程师的方式来测试网页。
   
3. **成本优势**：平均8~9分钟便可完成对一个应用15~20个功能点的测评，它可以24/7的去评估各种各样的应用。与此同时，却只花费0.26$每个网页，比真实人类评估便宜的多。

### 样例视频

（体现 输入需求，看到拆解的测试点，测试点的agent操作流程，测试结果）

## 安装

### 从零开始

```bash
# 创建conda环境
conda create -n appeval python=3.9
conda activate appeval

# 克隆仓库
git clone https://github.com/tanghaom/AppEvalPilot.git
cd AppEvalPilot

# 安装appeval
pip install -e .
```

### LLM配置

推荐配置：
- 编辑`config/config2.yaml`文件配置您的LLM模型
- 支持的模型：gpt-4o, gpt-4o-mini
- 确保在配置文件中设置您的`api_key`和`base_url`
- 对于其他多模态模型（例如 claude-3-5-sonnet-v2），请将它们添加到 `metagpt/provider/constant.py` 中的 MULTI_MODAL_MODELS

## 使用方法

### 基本命令

```bash
# 运行主程序
python main.py

# 启动服务
python scripts/server.py
```

### 重要参数


## 项目结构

```
AppEvalPilot/
├── main.py                           # 主程序入口
├── appeval/                          # 核心模块
│   ├── roles/                        # 角色定义
│   │   ├── appeval.py                # 自动化测试角色
│   │   └── osagent.py                # 操作系统代理
│   ├── actions/                      # 动作定义
│   │   ├── screen_info_extractor.py  # 屏幕信息提取
│   │   ├── test_generator.py         # 测试用例生成
│   │   └── reflection.py             # 反思
│   ├── tools/                        # 工具定义
│   │   ├── chrome_debugger.py        # 浏览器调试工具
│   │   ├── icon_detect.py            # 图标检测及描述工具
│   │   ├── device_controller.py      # 设备控制工具
│   │   └── ocr.py                    # ocr识别工具
│   └── utils/                        # 工具函数
├── scripts/                          # 脚本文件
│   ├── server.py                     # 部署服务脚本
│   └── test_server.py                # 测试服务脚本
├── data/                             # 数据文件
└── config/                           # 配置文件
```

## 贡献

我们欢迎对AppEvalPilot的贡献！如果您有问题、建议或想要贡献，请加入我们的Discord社区：[https://discord.gg/ZRHeExS6xv](https://discord.gg/ZRHeExS6xv)

## 引用

我们的论文将很快在arXiv上发布。请稍后查看引用信息。

## 许可证

此项目基于MIT许可证 - 详情请查看 LICENSE 文件
