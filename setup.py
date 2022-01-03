#!/usr/bin/env python3
"""Setup script"""

from pathlib import Path
import re

import setuptools

if __name__ == "__main__":

    # Read metadata from version.py
    with Path("compressed_embeddings/version.py").open(encoding="utf-8") as file:
        metadata = dict(re.findall(r'__([a-z]+)__\s*=\s*"([^"]+)"', file.read()))

    # Read description from README
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    _INSTALL_REQUIRES = [
        "numpy>=1.19.5",
        "scikit-learn>=0.24.2",
        "tensorflow>=2.6.2",
    ]

    _TEST_REQUIRE = ["pytest"]

    # Run setup
    setuptools.setup(
        name="compressed-embeddings",
        version=metadata["version"],
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Intended Audience :: Developers",
        ],
        long_description=long_description,
        long_description_content_type="text/markdown",
        description=long_description.split("\n")[0],
        author=metadata["author"],
        install_requires=_INSTALL_REQUIRES,
        tests_require=_TEST_REQUIRE,
        dependency_links=[],
        entry_points={"console_scripts": []},
        data_files=[(".", ["requirements.txt", "README.md"])],
        packages=setuptools.find_packages(),
        url="https://github.com/victor-paltz/compressed-embeddings",
    )
