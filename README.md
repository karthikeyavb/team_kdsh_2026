# TEAM_KDSH_2026
# Novel-Backstory Consistency Checker

A pipeline to check the consistency between novel text and character backstories using RAG (Retrieval Augmented Generation). This project is part of the Track A Submission for KDSH 2026.

## Overview

The tool extracts claims from a character's backstory, retrieves relevant segments from the novel's text using vector embeddings, and uses an LLM to verify if the claims are consistent with the novel. It aggregates these checks into a final consistency prediction (1 for Consistent, 0 for Inconsistent).

## Prerequisites

- **Python 3.8+**
- **OpenAI API Key**: Required for the LLM and embeddings.

## Installation

1.  **Clone the repository** (if you haven't already).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you are on Windows native, `pathway` might not be supported. The code includes a fallback to a simple internal pipeline.*

3.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_api_key_here
    ```

## Project Structure

```text
TEAM_KDSH_2026/
├── data/
│   ├── novels/         # Place novel .txt files here
│   └── backstories/    # Place backstory .txt or .json files here
├── outputs/            # Results will be saved here
├── src/                # Source code
│   ├── main.py         # Main pipeline logic
│   ├── config.py       # Configuration settings
│   └── ...
├── run.py              # Main entry point script
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Usage

### basic Usage

Run the consistency checker on all pairs in `data/novels` and `data/backstories`:

```bash
python run.py
```

### Advanced Options

You can customize the behavior using command-line arguments:

```bash
python run.py --novels-dir "path/to/novels" --backstories-dir "path/to/backstories" --output "outputs/my_results.csv"
```

### Command Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--novels-dir` | `data/novels` | Directory containing novel `.txt` files. |
| `--backstories-dir` | `data/backstories` | Directory containing backstory files. |
| `--output` | `outputs/results.csv` | Path for the output CSV file. |
| `--single` | `None` | Process only a specific story ID (e.g., `--single story1`). |
| `--chunk-size` | `1024` | Token size for text chunks. |
| `--top-k` | `10` | Number of chunks to retrieve per claim. |

### Example: Run for a single story

If you have `story1.txt` and `story1_backstory.txt` (or `.json`):

```bash
python run.py --single story1
```

## Outputs

The results are saved to the `outputs/` directory (or wherever specified by `--output`).

-   **`results.csv`**: A summary CSV containing:
    -   `Story ID`
    -   `Prediction` (1 = Consistent, 0 = Inconsistent)
    -   `Rationale` (Brief explanation)
-   **`results_detailed.json`**: A detailed JSON file with full claim analysis, retrieved chunks, and confidence scores.
