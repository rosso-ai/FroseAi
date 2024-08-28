from setuptools import find_packages, setup


def _requires_from_file():
    return open("requirements.txt").read().splitlines()


setup(
    name='froseai',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file(),
    url='https://github.com/rosso-ai-dataanalytics/FroseAi/',
    license='',
    author='Masahiko Hashimoto',
    author_email='',
    description=''
)
