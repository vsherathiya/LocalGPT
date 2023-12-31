
from setuptools import setup
import setuptools

with open('README.md', 'r', encoding='utf8-') as f:
    long_description = f.read()


__version = '0.0.0'

REPO_NAME = 'LocalGPT'
AUTHOR_USER_NAME = 'vsherathiya'
SRC_REPO = 'LocalGPT'
AUTHOR_EMAIL = 'viral.sherathiya@nexuslinkservices.in'

setup(
    name=SRC_REPO,
    version=__version,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='Local ChatGPT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=f'https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}',

    project_urls={
        'Bug_Tracker': f'https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues'
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src')
)