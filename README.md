# AppEvalPilot

## Project Overview

AppEvalPilot is an application evaluation automation tool designed to simplify the testing, evaluation, and analysis process of applications. By integrating various automation technologies, it helps developers and testers complete application evaluation work more efficiently.

## Key Features

- Automated test management
- Operating system agent integration
- Service deployment support
- Extensible evaluation framework

## Installation

### Prerequisites

- Python 3.9+
- Required dependencies

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/tanghaom/AppEvalPilot.git
cd AppEvalPilot

# Install appeval
pip install -e .

# Edit config/config2.yaml file to set up the LLM model
```

## Usage

### Basic Usage

```bash
# Run the main program
python main.py

# Start the service
python scripts/server.py
```

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

## Configuration

The project uses the `config/config2.yaml` file to store configuration information, including:

- LLM model
- base_url
- api_key

## License

This project is licensed under the MIT License - see the LICENSE file for details
