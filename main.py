from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from tools import load_csv_metadata, load_attributes, create_notebook
import json

load_dotenv()

llm = ChatAnthropic(model = "claude-haiku-4-5-20251001", max_tokens=8096)

# ─────────────────────────────────────────────────────────────────
# SYSTEM PROMPT 1 — EDA CODE GENERATION
# This tells Claude exactly what steps to always run,
# in what order, and what code to generate for each step.
# ─────────────────────────────────────────────────────────────────

EDA_SYSTEM_PROMPT = """You are an expert data analyst specializing in 
Exploratory Data Analysis (EDA).

You will receive CSV metadata and attribute descriptions.
Generate Python code and markdown cells for a complete Jupyter Notebook.

ALWAYS follow these steps in EXACTLY this order. 
Do NOT skip any step. Do NOT add extra steps.

---

STEP 1 — SETUP
Markdown cell: "# 1. EDA Report — [Dataset Name]"
Code cell must always contain:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    df = pd.read_csv('<csv_filename>')
    print("Dataset loaded successfully")
    print(f"Shape: {df.shape}")

---

STEP 2 — DATA OVERVIEW
Markdown cell: "## 2. Data Overview"
Code cell must always contain:
    print("=== First 5 Rows ===")
    display(df.head())
    print("\n=== Last 5 Rows ===")
    display(df.tail())
    print("\n=== Data Types ===")
    display(df.dtypes)
    print("\n=== Dataset Info ===")
    df.info()

---

STEP 3 — MISSING VALUE ANALYSIS
Markdown cell: "## 3. Missing Value Analysis"
Code cell must always contain:
    missing = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing = missing[missing['Missing Count'] > 0]
    print("=== Missing Values ===")
    display(missing)
    
    if not missing.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.show()
    else:
        print("No missing values found in the dataset")

---

STEP 4 — STATISTICAL SUMMARY
Markdown cell: "## 4. Statistical Summary"
Code cell must always contain:
    print("=== Numerical Columns Summary ===")
    display(df.describe())
    print("\n=== Categorical Columns Summary ===")
    display(df.describe(include='object'))

---

STEP 5 — DISTRIBUTION ANALYSIS
Markdown cell: "## 5. Distribution Analysis"
Always generate:
- Histograms for ALL numerical columns using:
    numerical_cols = df.select_dtypes(include=['int64','float64']).columns
    df[numerical_cols].hist(bins=30, figsize=(15, 10))
    plt.suptitle('Numerical Columns Distribution')
    plt.tight_layout()
    plt.show()

- Value counts bar charts for ALL categorical columns using:
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        df[col].value_counts().plot(kind='bar', color='steelblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

---

STEP 6 — CORRELATION ANALYSIS
Markdown cell: "## 6. Correlation Analysis"
Code cell must always contain:
    numerical_cols = df.select_dtypes(include=['int64','float64']).columns
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='coolwarm',
                    center=0,
                    square=True)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        print("\n=== Top Correlated Pairs ===")
        corr_pairs = corr_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs != 1.0]
        corr_pairs = corr_pairs.abs().sort_values(ascending=False)
        print(corr_pairs.head(10))
    else:
        print("Not enough numerical columns for correlation analysis")

---

STEP 7 — OUTLIER DETECTION
Markdown cell: "## 7. Outlier Detection"
Code cell must always contain:
    numerical_cols = df.select_dtypes(include=['int64','float64']).columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | 
                      (df[col] > Q3 + 1.5 * IQR)]
        print(f"{col}: {len(outliers)} outliers detected")
    
    plt.figure(figsize=(15, 8))
    df[numerical_cols].boxplot()
    plt.title('Boxplots — Outlier Detection')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

---

STEP 8 — DUPLICATE DETECTION
Markdown cell: "## 8. Duplicate Detection"
Code cell must always contain:
    duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicates}")
    print(f"Percentage: {(duplicates/len(df)*100).round(2)}%")
    
    if duplicates > 0:
        print("\n=== Sample Duplicate Rows ===")
        display(df[df.duplicated()].head())

---        

STEP 9 — SCATTERPLOTS
Markdown cell: "## 9. Scatterplots — Top Correlated Pairs"
Code cell must always contain:

    top_2 = corr_pairs.head(2).index.tolist()
    
    for col1, col2 in top_2:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=col1, y=col2, alpha=0.5)
        plt.title(f'{col1} vs {col2}')
        plt.tight_layout()
        plt.show()        

---

Use attribute descriptions to:
- Correctly label all chart titles and axes
- Identify and skip ID/identifier columns in analysis
- Correctly treat categorical codes (e.g. 1=Savings, 2=Current)
- Add context in markdown cells where relevant

CRITICAL RULES:
- Return ONLY a JSON array of cells
- Each cell must have exactly two keys: "cell_type" and "source"
- "cell_type" must be either "code" or "markdown"
- "source" must be a plain string with \\n for line breaks
- No markdown backticks anywhere
- No explanation outside the JSON
- Raw JSON array only"""

# ─────────────────────────────────────────────────────────────────
# SYSTEM PROMPT 2 — INSIGHTS GENERATION
# ─────────────────────────────────────────────────────────────────

INSIGHTS_SYSTEM_PROMPT = """You are a senior data analyst writing an
executive insights report.

You will receive dataset metadata and attribute descriptions.
Generate deep business relevant insights as Jupyter markdown cells.

ALWAYS cover these sections in EXACTLY this order:

---

SECTION 1 — EXECUTIVE SUMMARY
Markdown: "## 10. Executive Summary"
3-4 sentences summarising the dataset purpose and overall quality.

---

SECTION 2 — KEY FINDINGS
Markdown: "## 11. Key Findings"
Bullet points covering:
- Most significant patterns in the data
- Notable distributions or skews
- Strongest correlations found
- Any surprising observations

---

SECTION 3 — DATA QUALITY ASSESSMENT
Markdown: "## 12. Data Quality Assessment"
Cover:
- Missing data impact and which columns are affected
- Outlier severity and which columns are affected
- Duplicate records assessment
- Overall data quality rating: Good / Fair / Poor with justification

---

SECTION 4 — BUSINESS IMPLICATIONS
Markdown: "## 13. Business Implications"
Cover:
- What the patterns mean in a real business context
- Which columns are most analytically valuable
- Any segments or groups worth investigating further

---

SECTION 5 — RISK FLAGS
Markdown: "## 14. Risk Flags"
Cover:
- Anything anomalous or concerning
- Columns with high missing rates
- Potential data collection issues
- Any bias risks

---

SECTION 6 — RECOMMENDATIONS
Markdown: "## 15. Recommendations"
Cover:
- What data cleaning steps should follow this EDA
- Which columns should be DROPPED and why
  e.g. ID columns, high null columns, zero variance
- Which columns should be SPLIT and why
  e.g. combined columns like jobedu = job + education
- Which columns need TYPE CONVERSION
  e.g. categorical codes stored as integers
- Top 2 recommended scatterplots based on correlation
  with column names and expected insight
- What further analysis would be valuable

---

SECTION 7 — LIMITATIONS
Markdown: "## 16. Limitations"
Cover:
- What cannot be concluded from this dataset alone
- Assumptions made during analysis
- Sample size or time period concerns if relevant

---

CRITICAL RULES:
- Return ONLY a JSON array of markdown cells
- Each cell must have exactly: "cell_type": "markdown" and "source"
- No code cells in this response
- No markdown backticks
- Raw JSON array only"""


# ─────────────────────────────────────────────────────────────────
# MESSAGE BUILDERS
# ─────────────────────────────────────────────────────────────────

def build_eda_message(csv_path: str, metadata: dict,
                      attributes: dict) -> str:
    return f"""
CSV File Path: {csv_path}
Dataset Shape: {metadata['shape']}
Columns: {metadata['columns']}

Data Types:
{metadata['dtypes']}

Sample Rows (first 5):
{metadata['sample']}

Missing Values:
{metadata['nulls']}

Statistical Summary:
{metadata['describe']}

Attribute Descriptions:
{json.dumps(attributes, indent=2)}

Generate the complete EDA notebook following all 9 steps exactly.
"""


def build_insights_message(metadata: dict, attributes: dict) -> str:
    return f"""
Dataset Shape: {metadata['shape']}
Columns: {metadata['columns']}

Data Types:
{metadata['dtypes']}

Missing Values:
{metadata['nulls']}

Statistical Summary:
{metadata['describe']}

Attribute Descriptions:
{json.dumps(attributes, indent=2)}

Generate the business insights report covering all 7 sections exactly.
"""


# ─────────────────────────────────────────────────────────────────
# AGENT ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────

def run_eda_agent(csv_path: str, attributes_path: str,
                  output_path: str = "eda_report.ipynb"):

    # Step 1 — Load inputs locally
    print("Loading dataset metadata...")
    metadata = load_csv_metadata(csv_path)
    print(f"Dataset shape: {metadata['shape']}")

    print("Loading attribute descriptions...")
    attributes = load_attributes(attributes_path)
    print(f"Attributes loaded: {len(attributes)} columns described")

    all_cells = []

    # Step 2 — First Claude call: EDA code generation
    print("\nCalling Claude for EDA code generation...")
    eda_response = llm.invoke([
        SystemMessage(content=EDA_SYSTEM_PROMPT),
        HumanMessage(content=build_eda_message(
            csv_path, metadata, attributes))
    ])
    print("EDA code received.")

    # # Debug — print what Claude actually returned
    # print("\n=== RAW CLAUDE RESPONSE ===")
    # print(repr(eda_response.content))
    # print("=== END RESPONSE ===\n")

    eda_content = eda_response.content.strip()
    if eda_content.startswith("```"):
        eda_content = eda_content.split("```json")[-1]
        eda_content = eda_content.split("```")[0].strip()

    eda_cells = json.loads(eda_content)
    all_cells.extend(eda_cells)
    print(f"{len(eda_cells)} EDA cells generated.")

    # Step 3 — Second Claude call: Insights generation
    print("\nCalling Claude for business insights...")
    insights_response = llm.invoke([
        SystemMessage(content=INSIGHTS_SYSTEM_PROMPT),
        HumanMessage(content=build_insights_message(
            metadata, attributes))
    ])
    print("Insights received.")

    insights_content = insights_response.content.strip()
    if insights_content.startswith("```"):
        insights_content = insights_content.split("```json")[-1]
        insights_content = insights_content.split("```")[0].strip()

    insights_cells = json.loads(insights_content)
    all_cells.extend(insights_cells)
    print(f"{len(insights_cells)} insight cells generated.")

    # Step 4 — Build the notebook
    print("\nBuilding Jupyter notebook...")
    create_notebook(all_cells, output_path)
    print(f"\nDone! Open '{output_path}' in VS Code or Jupyter.")
    print("Run all cells to execute the full EDA on your dataset.")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT — update these before running
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    csv_path        = "bank_marketing_updated_v1.csv"       # your dataset
    attributes_path = "Attribute_details.xlsx"     # your attribute file
    output_path     = "eda_report.ipynb"    # output notebook name

    run_eda_agent(csv_path, attributes_path, output_path)

