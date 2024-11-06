from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
   name='ANCE',
   version='0.1.0',
   description='Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval',
   long_description=readme,
   long_description_content_type='text/markdown',
   url='https://github.com/microsoft/ANCE',
   classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
   license="MIT",
   packages=find_packages(),  # Automatically include all packages
   install_requires=[
        'transformers==2.3.0', 
        'pytrec-eval',
        'faiss-cpu',
        'wget',
    ],
   python_requires='>=3.6',
)

