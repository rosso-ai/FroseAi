from setuptools import find_packages, setup


def _requires_from_file():
    return open("requirements.txt").read().splitlines()


setup(
    name='froseai',
    version='0.0.2',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file(),
    url='https://github.com/rosso-ai-dataanalytics/FroseAi/',
    license='Apache License',
    author='Masahiko Hashimoto',
    entry_points={
        "console_scripts": [
            "frose_run = froseai.demo.runner:run",
        ]
    },
    author_email='',
    description=''
)
