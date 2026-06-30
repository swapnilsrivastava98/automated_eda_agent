# Automated EDA Agent

A LangChain + Claude powered agent that analyzes CSV files and automatically generates comprehensive Jupyter notebooks with statistical summaries, visualizations, data quality checks, and insights.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/swapnilsrivastava98/automated_eda_agent.git
cd automated_eda_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp sample.env .env
```
Add your `ANTHROPIC_API_KEY` to `.env`.

## Usage

Place your CSV file and an attributes file (`.xlsx`, with column descriptions) in the project directory. Update the file paths at the bottom of `main.py`:

```python
csv_path = "your_dataset.csv"
attributes_path = "Attribute_details.xlsx"
output_path = "eda_report.ipynb"
```

Then run:
```bash
python main.py
```

The agent generates a complete Jupyter notebook (EDA + business insights) at the specified output path.

## Features

- Data profiling (types, missing values, memory usage)
- Statistical analysis (distributions, correlations, outliers)
- Visualizations (histograms, heatmaps, boxplots, scatterplots)
- Duplicate and outlier detection
- Business insights report (key findings, data quality, risk flags, recommendations)
- Auto-generated Jupyter notebook with clean, reproducible code

## Requirements

Python 3.10+, pandas, langchain, jupyter, matplotlib, seaborn, numpy, openpyxl

## License

MIT
