# Wisconsin Lakes Park Selection Analysis

A reproducible data science project demonstrating how to create publication-ready documents from Jupyter notebooks using Quarto.

## Project Overview

This repository contains a comprehensive data analysis of Wisconsin lakes, serving as a demonstration of reproducible data science practices. The project showcases how to transform Jupyter notebooks into multiple publication formats (HTML, PDF, and DOCX) while maintaining reproducibility and professional formatting.

## Environmental + Tech Stack

- **Python 3.13** - Core programming language
- **Jupyter** - IDE/Interactive computing environment
- **Quarto** - Publishing system for scientific and technical writing
- **Key Python Libraries**:
  - pandas - Data manipulation and analysis
  - seaborn/matplotlib - Data visualization
  - geopandas - Geospatial data handling
  - scikit-learn - Machine learning components
  - numpy - Numerical computing

## Repository Structure

```
.
├── ParkSiteSelection.ipynb    # Main analysis notebook
├── _quarto.yml                # Quarto configuration
├── references.bib             # Bibliography file
├── styles.css                 # Custom CSS styles
├── figures/                   # Generated figures
└── original_data/             # Source data files
    ├── Lakes_Original.csv
    └── gigsheet-counties.csv
```

## Reproducibility Features

### Quarto Integration

The project uses Quarto for document generation, supporting:
- Multiple output formats (HTML, PDF, DOCX)
- Automatic table of contents
- Code folding and tools
- Bibliography management
- Custom styling
- Watermarking for draft versions

### Code Organization

- Analysis contained in a single Jupyter notebook for clarity
- Code cells include YAML-style metadata for Quarto processing
- Figures include labeling and cross-referencing
- Data processing steps documented

## Getting Started

1. Clone this repository
2. Install Full Anaconda Distribution [anaconda.com](https://www.anaconda.com/download)
3. Install Quarto from [quarto.org](https://quarto.org)
4. Also install Quarto's tinytex `quarto install tinytex --update-path`
5. Open `ParkSiteSelection.ipynb` in Jupyter

## Rendering Documents

To render the documents in different formats:

```bash
# Preview HTML With
quarto preview ParkSiteSelection.ipynb

# Render HTML
quarto render ParkSiteSelection.ipynb --to html

# Render PDF
quarto render ParkSiteSelection.ipynb --to pdf

# Render Word document
quarto render ParkSiteSelection.ipynb --to docx
```

## Data Sources

- Primary dataset: Wisconsin Department of Natural Resources lake data (16,711 lakes)
- Supporting dataset: Wisconsin county information

## Author

Adam Ross Nelson, JD PhD  
University of Wisconsin - Madison  
Contact: arnelson3@wisc.edu

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{nelson2025wisconsin,
  author = {Nelson, Adam Ross},
  title = {Applied Reproducible Data Science Processes: A demonstration using data from Wisconsin lakes},
  year = {2025},
  institution = {University of Wisconsin - Madison}
}
