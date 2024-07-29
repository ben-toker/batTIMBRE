# batTIMBRE
Welcome to the batTIMBRE project! This repository is dedicated to applying the [TIMBRE architecture (developed by Professor Gautam Agarwal)](https://github.com/beatLaboratory/TIMBRE) to neuropixel probe data on bats. The primary goal of this project is to verify whether Local Field Potential (LFP) recordings encode positional information in bats. By analyzing both positional and neural data, we aim to gain insights into the neural underpinnings of bat navigation and spatial awareness.

### Bat flight paths!
![Bat Flight Paths by Cluster](graphs/allflights.png)


## Setting up the development (virtual) environment

We need to create a virtual environment (venv) to handle the odd dependencies of this project. Since we're trying to get two different projects to talk to eachother nicely, this is easily solved with a venv:

```bash
python -m venv venv
```

### To activate the environment:

Windows:
```bash
venv\Scripts\activate
```
macOS/Linux:
```bash
source venv/bin/activate
```
Install dependencies using ```setuptools```
```bash
pip install -e .
```

Doing it this way ensures that we can easily establish imports between folders and files using the toplevel structure.
