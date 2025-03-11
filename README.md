# AppEvalPilot

## Introduction

AppEvalPilot is an automated evaluation tool for assessing software application functionality completeness. It's designed to work with desktop applications, mobile/app applications, and web applications.

AppEvalPilot can help you automatically evaluate any application without human intervention, saving time and resources while maintaining high accuracy.

In evaluations with 2000+ test cases, AppEvalPilot has demonstrated high correlation with human expert judgments (Pearson correlation coefficient across all versions is 0.9249, and the average Spearman correlation coefficient is 0.9021).

### Features

1. **Ease of Use**: One codebase to evaluate desktop applications, mobile/app applications, and web applications.
   
2. **Robust and Reliable Dynamic Assessment**: Unlike other benchmarks that rely on static evaluation methods, AppEvalPilot simulates the workflow of web testing engineers to test applications.
   
3. **Cost-Effective**: AppEvalPilot can complete the evaluation of 15-20 functional points in an application in just 8-9 minutes on average. It can operate 24/7 to evaluate various applications while costing only $0.26 per webpage - significantly cheaper than human evaluation.

### Sample Videos

(Videos demonstrating the input requirements, breakdown of test points, agent operation workflow for test points, and test results)

## Installation

### From Scratch

```bash
# Create a conda environment
conda create -n appeval python=3.10
conda activate appeval

# Clone the repository
git clone https://github.com/tanghaom/AppEvalPilot.git
cd AppEvalPilot

# Install appeval
pip install -e .
```

### LLM Configuration

Recommended configuration:
- Edit `config/config2.yaml` to configure your LLM model
- Supported models: gpt-4o, gpt-4o-mini 
- Make sure to set your `api_key` and `base_url` in the configuration file
- For other multimodal models(eg. claude-3-5-sonnet-v2), please add them to `metagpt/provider/constant.py` in MULTI_MODAL_MODELS
## Usage

### Basic Commands

```bash
# Run the main program
python main.py

# Start the service
python scripts/server.py
```

### Important Parameters


## Project Structure

```
AppEvalPilot/
├── main.py                           # Main program entry
├── appeval/                          # Core modules
│   ├── roles/                        # Role definitions
│   │   ├── appeval.py                # Automated testing role
│   │   └── osagent.py                # Operating system agent
│   ├── actions/                      # Action definitions
│   │   ├── screen_info_extractor.py  # Screen information extraction
│   │   ├── test_generator.py         # Test case generation
│   │   └── reflection.py             # Reflection
│   ├── tools/                        # Tool definitions
│   │   ├── chrome_debugger.py        # Browser debugging tool
│   │   ├── icon_detect.py            # Icon detection and description tool
│   │   ├── device_controller.py      # Device control tool
│   │   └── ocr.py                    # OCR recognition tool
│   └── utils/                        # Utility functions
├── scripts/                          # Script files
│   ├── server.py                     # Service deployment script
│   └── test_server.py                # Service testing script
├── data/                             # Data files
└── config/                           # Configuration files
```

## Contribution

We welcome contributions to AppEvalPilot! If you have questions, suggestions, or would like to contribute, please join our Discord community: [https://discord.gg/ZRHeExS6xv](https://discord.gg/ZRHeExS6xv)

## Citation

Our paper will be available on arXiv soon. Please check back for citation information.

## License

This project is licensed under the MIT License - see the LICENSE file for details
