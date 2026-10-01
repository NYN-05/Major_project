from setuptools import setup, find_packages

setup(
    name="deepfake-rppg",
    version="1.0.0",
    packages=find_packages(include=["rppg", "rppg.*", "visual", "visual.*", "quantum", "quantum.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.26.4",
        "scipy>=1.11.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "opencv-python>=4.8.0,<5.0.0",
        "mediapipe>=0.10.9",
        "pandas>=2.0.0",
        "scikit-image>=0.26.0",
        "pennylane>=0.36.0",
        "pennylane-lightning>=0.36.0",
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        "torchaudio==2.5.1+cu121",
        "ultralytics==8.3.0",
        "xgboost>=2.0.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "streamlit>=1.30.0",
    ],
    extras_require={
        # ORT 1.20.2 supports the cuDNN 9 runtime bundled with PyTorch 2.5.1.
        "gpu": ["onnxruntime-gpu==1.20.2"],
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)