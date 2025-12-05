from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "ReadMe.md").read_text()

setup(
  name='grandlantern',
  version='0.0.21',
  author='grandyarl',
  author_email='grand_yarl@mail.com',
  url='https://github.com/grand-yarl/grand_lantern',
  description='This is the small library for deep learning.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages=find_packages(),
  classifiers=[
    'Programming Language :: Python :: 3.9',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  keywords='deeplearn',
  project_urls={
    'GitHub': 'https://github.com/grand-yarl/grand_lantern'
  },
  python_requires='>=3.6'
)