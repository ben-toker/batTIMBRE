### Setting up the development (virtual) environment

We need to create a virtual environment (venv) to handle the odd dependencies of this project. Since we're trying to get two different projects to talk to eachother nicely, this is easily solved with a venv:

```bash
python -m venv venv
```

To activate the environment:
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