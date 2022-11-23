import setuptools

setuptools.setup(
    include_package_data = True,
    name = 'ServierMoleculePropertyPrediction',
    version = '1.0.0',
    licence = 'MIT',
    description = 'Servier Molecule Properties Prediction python module',
    url = 'https://github.com/BasileKoffiDataScientist/data_science_technical_test/blob/main',
    author = 'Basile Koffi',
    author_email = 'koffibasile@gmail.com',
    # I can parse here the requirement.txt file. But I'm in Apha version. So I keet it simple
    install_requires = ["pandas", "rdkit", "transformers"],
    long_description = "This application predict basic molecule's properties, from its fingerprint features and smile string, using deep learning",
    long_description_content_type = "text/markdown",
    classifiers = [
        "Programming Language  :: Python ::3",
        "Operating System  :: OS Independant",
    ],
    entry_points='''
        [console_scripts]
        predmol = app.cli:main
        ''',
)