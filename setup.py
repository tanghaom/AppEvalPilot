from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

extras_require = {
    "ultra": [
        "torch==2.5.1",
        "torchvision==0.20.1",
        "tensorflow",
        "tf_slim",
        "transformers",
        "modelscope",
        "ultralytics",
    ]
}

setup(
    name="appeval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required,
    extras_require=extras_require,
    author="xxx",
    author_email="xxx",
    description="一个应用程序评估工具",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tanghaom/AppEvalPilot",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
