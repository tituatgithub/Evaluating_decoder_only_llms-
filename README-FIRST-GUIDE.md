# MUST READ INSTRUCTIONS BEFORE WORKING IN THE REPOSITORY

## Technical Notes
1. Please setup separate python virtual environment using below things and then start working.
2. Please use linux (recommended).

## Working notes
1. Please name all files and folders in lowercase and if you are using multiple words, then separate them using underscores.

Example:    
1. File names, `this_is_a_valid_python_file.py` and `my_model.py` is fine but `This-is-Not a valid file.py` is not valid.
2. Folder names, `this_is_valid` but `thisIsNot`.

## 1. Create and activate a python virtual environment [Use linux (Recommended)]

### 1. Create a virtual environment as:

- One time setup only:
```
sudo apt install python3-venv
```

```bash
python3 -m venv .venv
```

or in windows:
```bash
python3.exe -m venv .venv
```

### 2. Activate it

```bash
source .venv/bin/activate
```

or in windows:
```bash
.\Scripts/bin/activate.ps1
```

and if the upper one doesn't works then use chatgpt to troubleshoot. Its easy:) You got it!

## Then install all the dependencies

```bash
pip install -r requirements.txt
```
