# LLM-adaptation-techniques.-Evaluating-RAG-models
## Code Structure and Dependencies

### `functions.py`
- The file `functions.py` contains auxiliary functions that are used in the notebook.
- These functions are imported directly into the notebook, and they provide the necessary functionality for various parts of the code.

### Python Version
- The code is designed to work with **Python 3.10.15**.
- It is important to ensure that the correct Python version is installed to avoid compatibility issues.

### Installing Dependencies
To reproduce the environment used in this project, follow these steps:

1. **Ensure Python 3.10.15 is Installed**:

2. **Install Dependencies**:
   - The project dependencies are listed in the `requirements.txt` file.
   - Use the following command to install them:
     ```bash
     pip install -r requirements.txt
     ```

### Important Note
- Ensure that both the notebook file and `functions.py` are **in the same folder** for seamless imports.


## Flags for Data Generation and Handling

The following flags are used in the code to control the behavior of the answer generation and data handling processes:

### `Generating` Flag
- Determines whether the program generates all answers and databases or loads them from pre-existing files on the disk.
- **Options**:
  - `True`: The program generates all answers and creates databases from scratch.
  - `False`: The program loads previously generated answers and databases from the disk, skipping the generation step.
- **Use Case**: Set this to `False` if the data has already been generated in a previous run and you want to save computation time.

### `Saving` Flag
- Determines whether the program saves the newly generated answers and databases to the disk for future use.
- **Options**:
  - `True`: The generated answers and databases will be saved to the disk.
  - `False`: The generated answers and databases will not be saved.
- **Use Case**: Set this to `False` if you only need the data for a single run and are not interested in saving it for future use. This is useful for temporary experiments or one-off tests.

---