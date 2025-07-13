# Analysis of Research Trends on arXiv

**A Case Study for the IU - Internationale Hochschule Module 'Unsupervised Learning and Feature Engineering in Machine Learning'**

## Overview

This project presents Python Notebook Files belonging to a case study on the analysis of scientific research papers from the arXiv repository. The primary goal is to process, analyze, and cluster over a decade of publication data to uncover underlying thematic structures and identify emerging research topics.

The analysis pipeline involves several stages:
*   **Data Ingestion and Cleaning:** Processing a large JSON dataset of arXiv metadata, cleaning it, and transforming it into an efficient format.
*   **Feature Engineering:** Extracting features from the raw data, including metadata (e.g., author count), categorical information (research domains), and text-based features derived from paper abstracts using TF-IDF.
*   **Clustering and Dimensionality Reduction:** Applying a sophisticated pipeline involving PCA, UMAP, and HDBSCAN to group papers into clusters based on their features.
*   **Trend Analysis:** Analyzing the temporal evolution of these clusters to identify which research areas are gaining popularity and are considered "emerging clusters".


## Dataset

This project uses the [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv/versions/165) from Kaggle, which contains metadata for papers submitted to arXiv.

To run the notebooks, you must first download the dataset. The file is named `arxiv-metadata-oai-snapshot.json`.

**Action Required:**
1.  Download the dataset from the link above.
2.  Place the `arxiv-metadata-oai-snapshot.json` file in the root directory of this project.
3.  The first notebook in the pipeline expects this exact filename (First notebook: 'Transform_to_parquet.ipynb').

## Setup

To set up your local environment to run this project, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Thymin3/Research-Trend-Analysis.git
    cd Research-Trend-Analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    The project depends on several Python libraries. You can install them using pip:
    ```bash
    pip install pandas pyarrow pandarallel numpy scikit-learn umap-learn hdbscan matplotlib seaborn plotly wordcloud nltk inflect
    ```

4.  **Download NLTK data:**
    The feature engineering notebook requires NLTK data for text processing ('Feature_Engineering.ipynb'). Run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

## Execution Workflow

The analysis is structured across several Jupyter Notebooks that should be executed in the following order. Each notebook performs a specific part of the pipeline and generates intermediate files that are used by subsequent notebooks.

1.  **`Transform_to_parquet.ipynb`**
    *   **Purpose:** Converts the raw `arxiv-metadata-oai-snapshot.json` data into the more memory-efficient Apache Parquet format.
    *   **Input:** `arxiv-metadata-oai-snapshot.json`
    *   **Output:** `arxiv.parquet`

2.  **`Categories.ipynb`**
    *   **Purpose:** Parses the official arXiv category taxonomy text to create a structured and clean reference table of research domains, areas, and sub-areas
    *   **Input:** None.
    *   **Output:** `categories.parquet`

3.  **`Cleaning.ipynb`**
    *   **Purpose:** Performs major data cleaning tasks. This includes dropping irrelevant columns, removing records without a DOI, handling duplicates, filtering for papers published since 2015, and standardizing category information using the `categories.parquet`
    *   **Input:** `arxiv.parquet`, `categories.parquet`
    *   **Output:** `arxiv_cleaned.parquet`

4.  **`Feature_Engineering.ipynb`**
    *   **Purpose:** Engineers a set of features from the cleaned data. It calculates author counts, creates binary columns for research domains and areas, and performs text processing on abstracts to extract and binarize the most relevant keywords
    *   **Input:** `arxiv_cleaned.parquet`
    *   **Output:** `checkpoint_with_keywords.parquet`, `checkpoint_variables.pkl`

5.  **`Clustering.ipynb`**
    *   **Purpose:** Applies the core machine learning pipeline. It uses the engineered features to perform dimensionality reduction and clustering, visualizes the results, and runs a temporal analysis to characterize the clusters and identify emerging research topics
    *   **Input:** `checkpoint_with_keywords.parquet`, `checkpoint_variables.pkl`
    *   **Output:** `results_df_with_clusters.parquet`, various plots and analysis tables

---
*   **`Data_exploration.ipynb` (Optional)**
    *   This notebook contains initial exploratory data analysis and various visualizations created after the cleaning step. It is not a required part of the main feature engineering and clustering pipeline.

## Methodology

The project employs a multi-stage methodology to identify research clusters and trends:

*   **Feature Engineering:** A combination of metadata (author count), one-hot encoded categorical features (domains, areas), and text-based features are created. Keywords are extracted from abstracts using a domain-specific TF-IDF approach to ensure relevance.

*   **Clustering Pipeline:** A pipeline is used to handle the high-dimensional feature space and uncover patterns:
    1.  **StandardScaler:** All features are scaled to have a mean of 0 and a standard deviation of 1, ensuring that each feature contributes equally to the analysis.
    2.  **PCA (Principal Component Analysis):** An initial, linear dimensionality reduction is performed to reduce noise and computational complexity for the next step.
    3.  **UMAP (Uniform Manifold Approximation and Projection):** A non-linear dimensionality reduction technique is used to find the underlying structure of the data, preserving the relationships between data points in a lower-dimensional space.
    4.  **HDBSCAN (Hierarchical Density-Based Clustering):** This density-based algorithm is applied to the UMAP-transformed data. It can identify clusters of varying shapes and sizes and automatically flags data points that do not belong to any cluster as noise.

*   **Trend Analysis:** After clustering, the temporal distribution of papers within each cluster is analyzed. By comparing the proportion of a cluster's publications in a recent period to a baseline period, "emerging" topics are identified that are gaining traction in the research community.
