from setuptools import setup, find_packages

setup(name='sklearn_pipeline_enhancements',
      version='0.1',
      description='experimental enchancements to sklearn pipelines',
      author='kgoetsch',
      author_email='goetscher@gmail.com',
      packages=find_packages(),
      zip_safe=False, install_requires=['numpy', 'pandas', 'patsy', 'sklearn'])
