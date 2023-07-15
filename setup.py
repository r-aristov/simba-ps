from setuptools import setup, find_packages

setup(
    name="simba_ps",
    version="1.0.0",
    author="Roman Aristov",
    description="Fast deterministic all-Python Lennard-Jones particle simulator that utilizes Numba for GPU-accelerated computation.",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.6",
        "numba>=0.55.1",
        "pillow>=9.2.0",
        "pygame>=2.1.0",
        "cuda-python>=11.4"
    ],
    entry_points={
        "console_scripts": [
            "simba-impact = simba_examples.impact:main",
            "simba-vortex = simba_examples.vortex:main",
            "simba-hexcells = simba_examples.hexcells:main",
            "simba-chemistry-101 = simba_examples.chemistry_101:main",
            "simba-chemistry-201 = simba_examples.chemistry_201:main",
            "simba-deterministic-fall = simba_examples.deterministic_fall:main",
            "simba-distillation = simba_examples.distillation:main",
        ]
    }
)

