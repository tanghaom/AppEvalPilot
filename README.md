# AppEvalPilot

## Introduction

Welcome to the AppEvalPilot project, a cutting-edge automated evaluation framework designed to comprehensively assess software application functionalities across an array of platforms. Tailored for versatility, this framework adeptly handles the evaluation of desktop, mobile, and web-based applications under a unified methodology. 

AppEvalPilot's fully automated process operates without manual intervention, streamlining your workflow while significantly cutting costs. By leveraging our framework, you not only accelerate the evaluation process but also achieve superior accuracy in assessment outcomes. Ideal for developers and QA teams looking to enhance efficiency and quality in their testing procedures, AppEvalPilot stands out as a reliable solution for comprehensive, precise, and efficient application assessments. Join us in advancing software evaluation with AppEvalPilot.

### Features

1. **Cross-Platform Compatibility**: A unified codebase facilitating evaluation across desktop applications, mobile applications, and web-based interfaces.
   
2. **Methodologically Robust Dynamic Assessment**: In contrast to conventional benchmarks employing static evaluation methodologies, AppEvalPilot replicates the systematic workflow of professional testing engineers to conduct thorough application evaluation.
   
3. **Resource Efficiency**: AppEvalPilot completes comprehensive evaluation of 15-20 functional components within an application in approximately 8-9 minutes. The system operates continuously (24/7) to evaluate diverse applications at a cost of $0.26 per webpage—substantially more economical than human-conducted evaluations.

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
# Enhanced version of appeval with OCR and icon detection capabilities
pip install -e .[ultra]
```

### LLM Configuration

- Edit `config/config2.yaml` to configure your LLM model
- Recommended models: claude-3-5-sonnet-v2
- Ensure appropriate configuration of `api_key` and `base_url` parameters in the configuration file
- For integration of additional multimodal models (e.g., Qwen2.5-VL-72B), add the corresponding model identifiers in [`metagpt/provider/constant.py`](https://github.com/geekan/MetaGPT/blob/79390a28247dbfaf8097d3bcd6e6f23b56e9e444/metagpt/provider/constant.py#L34)

## Usage

### Basic Commands

```bash
# Run the main program
python main.py
```

```bash
# Run OSagent
python scripts/run_osagent.py
```

```bash
# Start the service
python scripts/server.py
```

### Important Parameters


## Project Structure

```
AppEvalPilot/
├── main.py                           # Main program entry
├── setup.py                          # Package setup script
├── appeval/                          # Core modules
│   ├── roles/                        # Role definitions
│   │   ├── test_runner.py            # Automated testing role
│   │   └── osagent.py                # Operating system agent
│   ├── actions/                      # Action definitions
│   │   ├── screen_info_extractor.py  # Screen information extraction
│   │   ├── case_generator.py         # Test case generation
│   │   └── reflection.py             # Reflection and analysis
│   ├── tools/                        # Tool definitions
│   │   ├── chrome_debugger.py        # Browser debugging tool
│   │   ├── icon_detect.py            # Icon detection and description tool
│   │   ├── device_controller.py      # Device control tool
│   │   └── ocr.py                    # OCR recognition tool
│   ├── prompts/                      # Prompt templates
│   │   ├── test_runner.py            # Application evaluation prompts
│   │   └── osagent.py                # OS agent prompts
│   ├── utils/                        # Utility functions
│   │   ├── excel_json_converter.py   # Excel and JSON format conversion utilities
│   │   └── window_utils.py           # Window control and browser automation utilities
│   └── __init__.py                   # Package initialization
├── scripts/                          # Script files
│   ├── server.py                     # Service deployment script
│   └── test_*.py                     # Various component test scripts
├── data/                             # Data files
├── config/                           # Configuration files
│   └── config2.yaml.example          # Example configuration template
└── work_dirs/                        # Working directories for runtime data
```

## Contribution

Contributions to AppEvalPilot are welcomed by the research community. For inquiries, suggestions, or potential collaborations, please join our Discord community: [https://discord.gg/ZRHeExS6xv](https://discord.gg/ZRHeExS6xv)

## Citation

The corresponding research paper will be available on arXiv in the near future. Please refer back for citation information.

## License

This project is distributed under the MIT License - refer to the LICENSE file for comprehensive details.
