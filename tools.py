import pandas as pd
import nbformat
import json
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


# ─────────────────────────────────────────────
# FUNCTION 1: LOAD CSV METADATA
# ─────────────────────────────────────────────
# Purpose: Read the dataset CSV and extract only
# the metadata (not the full raw data) to send
# to Claude. This keeps token usage low and cost
# efficient regardless of dataset size.

def load_csv_metadata(csv_path: str) -> dict:
    """Extract metadata from CSV without sending raw data to Claude"""
    
    df = pd.read_csv(csv_path)
    
    return {
        # Number of rows and columns e.g. (1000, 12)
        "shape": str(df.shape),
        
        # List of all column names
        "columns": list(df.columns),
        
        # Data type of each column (int, float, object etc.)
        "dtypes": df.dtypes.to_string(),
        
        # First 10 rows so Claude understands data format
        "sample": df.head(10).to_string(),
        
        # Count of missing/null values in columns with null values
        "nulls": df.isnull().sum()[df.isnull().sum() > 0].to_string(),
        
        # Statistical summary - mean, std, min, max etc.
        # include='all' covers both numerical and categorical
        "describe": df.describe(include='all').to_string()
    }


# ─────────────────────────────────────────────
# FUNCTION 2: LOAD ATTRIBUTE DESCRIPTIONS
# ─────────────────────────────────────────────
# Purpose: Read the attribute file (Excel or CSV)
# that describes what each column means. This
# helps Claude correctly interpret ambiguous or
# cryptic column names like 'cd_type' or 'flag_1'.
# Expects two columns in the file:
#   - column_name  : matches your dataset columns
#   - description  : plain English explanation

def load_attributes(attributes_path: str) -> dict:
    """Load attribute descriptions from Excel or CSV file"""
    
    # Handle both Excel and CSV formats
    if attributes_path.endswith('.xlsx') or \
       attributes_path.endswith('.xls'):
        df = pd.read_excel(attributes_path)
    else:
        df = pd.read_csv(attributes_path)

    df.columns = ['column_name', 'description']

    # Drop rows where column_name is null (empty rows)
    df = df.dropna(subset=['column_name'])

    # Fill missing descriptions with a placeholder
    df['description'] = df['description'].fillna('No description provided')

    # Strip extra whitespace from both columns
    df['column_name'] = df['column_name'].str.strip()
    df['description'] = df['description'].str.strip()    
    
    # Convert to a dictionary for easy reading
    # e.g. {"cust_id": "Unique customer ID", 
    #        "trx_amt": "Transaction in INR"}
    return dict(zip(df['column_name'], df['description']))


# ─────────────────────────────────────────────
# FUNCTION 3: CREATE JUPYTER NOTEBOOK
# ─────────────────────────────────────────────
# Purpose: Take the list of cells returned by
# Claude (as JSON) and build an actual .ipynb
# Jupyter notebook file from them.
#
# Claude returns cells in this format:
# [
#   {"cell_type": "markdown", "source": "# EDA Report"},
#   {"cell_type": "code", "source": "import pandas as pd"},
#   ...
# ]
#
# nbformat is the official Python library for
# creating and editing Jupyter notebook files.

def create_notebook(cells_data: list, output_path: str):
    """Build a .ipynb file from Claude's generated cells"""
    
    # Create a blank notebook object
    nb = new_notebook()
    
    cells = []
    
    for cell in cells_data:
        
        if cell["cell_type"] == "code":
            # Code cells contain executable Python code
            cells.append(new_code_cell(cell["source"]))
            
        elif cell["cell_type"] == "markdown":
            # Markdown cells contain headings, text,
            # insights, explanations etc.
            cells.append(new_markdown_cell(cell["source"]))
    
    # Attach all cells to the notebook
    nb.cells = cells
    
    # Write the notebook to disk as a .ipynb file
    with open(output_path, "w") as f:
        nbformat.write(nb, f)
    
    print(f"Notebook saved: {output_path}")