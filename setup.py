from setuptools import setup, find_packages

setup(
    name='tf-semiseg-dataloader',
    version='0.0.1',
    packages=find_packages(),
    license='MIT',
    description='Custom dataloader for semiseg segmentation dataset',
    long_description=open('README.md').read(),
    long_description_content_type = "text/markdown",
    install_requires=['tensorflow', 'numpy'],
    url='https://github.com/SandroRybarik/tf-semiseg-dataloader',
    author='SandroRybarik',
    author_email='rybarik@fisherio.com'
)