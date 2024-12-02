from setuptools import setup, find_packages

setup(
    name="vietnamese-summarizer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.21.0",
        "gradio>=3.0.0",
        "sentencepiece",
        "deep_translator",
    ],
)