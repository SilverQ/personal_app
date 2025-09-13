# Gemini Project Context

This document provides context for the Gemini AI, helping it understand the project's structure, conventions, and goals.

## 1. Project Overview

*   **Goal:** To develop a comprehensive personal finance application that includes features for stock analysis, currency tracking, and potentially trading.
*   **Core Features:**
    *   Stock analysis using various techniques (e.g., Mahalanobis distance).
    *   Currency exchange rate tracking and visualization.
    *   Integration with external APIs (e.g., Kiwoom REST API) for financial data.
    *   A web-based user interface built with Streamlit.

## 2. Tech Stack

*   **Primary Language:** Python
*   **UI Framework:** Streamlit
*   **Data Analysis/Manipulation:** pandas, numpy
*   **Machine Learning/Statistics:** scikit-learn
*   **API Interaction:** requests, potentially other specific API client libraries
*   **Package Manager:** pip (as indicated by `requirements.txt`)

## 3. Project Structure

*   `main_app.py`: The main entry point for the Streamlit application.
*   `apps/`: A directory containing different modules or "apps" that are part of the main application.
    *   `stock_analysis_app.py`: App for stock analysis.
    *   `currency_app.py`: App for currency tracking.
*   `data/`: Contains data files like CSVs and images.
*   `requirements.txt`: Lists the Python dependencies for the project.
*   `config.ini`: Configuration file for storing settings and API keys.
*   `*.py`: Various scripts for tasks like crawling (`crawling.py`), data processing (`data_processor.py`), and updating stock lists (`update_stock_list.py`).

## 4. Key Commands

*   **Running the main application:** `streamlit run main_app.py`
*   **Installing dependencies:** `pip install -r requirements.txt`
*   **Running standalone scripts:** `python <script_name>.py` (e.g., `python update_stock_list.py`)

## 5. Coding Conventions

*   **Style:** Adhere to PEP 8 for Python code style.
*   **Imports:** Use absolute imports where possible.
*   **Configuration:** Store sensitive information like API keys in `config.ini` and do not commit it to version control.
*   **Data:** Store larger data files in the `data/` directory.
*   **Modularity:** Keep different functionalities in separate files within the `apps/` directory to ensure the main application remains clean and organized.

## 6. Code Modification Guidelines

To minimize errors and maintain code quality, please follow these guidelines when modifying code:

1.  **Understand the Context:** Before making any changes, thoroughly analyze the surrounding code, its purpose, and existing patterns.
2.  **Prioritize Testing:**
    *   Search for and run existing tests to ensure your changes do not break other parts of the application.
    *   If tests do not exist for the code you are modifying, consider adding simple tests to verify the functionality of your changes.
3.  **Linting and Style:**
    *   After making changes, run a linter (if available) to check for style inconsistencies and potential errors.
    *   Ensure all new code adheres to the PEP 8 style guide.
4.  **Incremental Changes:**
    *   Make small, incremental changes rather than large, complex ones.
    *   Test after each small change to catch errors early.
5.  **Dependency Management:**
    *   If you add a new library, ensure it is added to the `requirements.txt` file.
6.  **Clarification:**
    *   If a request is ambiguous or could have multiple interpretations, ask for clarification before proceeding with implementation.