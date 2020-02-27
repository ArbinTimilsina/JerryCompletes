from setuptools import setup

setup(
    name='jerry_completes',
    version='0.1',
    maintainer='Arbin Timilsina',
    maintainer_email='arbin.timilsina@gmail.com',
    platforms=['any'],
    description='Let Jerry Seinfeld complete an incomplete sentence.',
    packages=['jerry_completes'],
    include_package_data=True,
    install_requires=[
        'torch', 'transformers==2.5.0', 'nltk', 'tqdm', 'flask', 'ftfy'
    ],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        'console_scripts': [
            'train-jerry = jerry_completes.cli_train_jerry:main',
            'serve-jerry = jerry_completes.cli_server:main',
        ]
    },
)
