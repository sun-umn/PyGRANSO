import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygranso",
    version="1.0.0-dev3",
    author="Tim Mitchell and Buyun Liang",
    author_email="liang664@umn.edu, tim@timmitchell.com",
    description="PyGRANSO: A PyTorch-enabled port of GRANSO with auto-differentiation",
    keywords=['deep learning', 'machine learning', 'optimization software', 'mathematical software'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sun-umn/PyGRANSO",
    project_urls={
        "Bug Tracker": "https://github.com/sun-umn/PyGRANSO/issues",
    },
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.9",
)