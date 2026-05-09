# Wisconsin Lakes Park Selection Analysis

Reproducible data science demonstrating how to create publication-ready documents from Jupyter notebooks using Quarto.

## Project Overview

This repository contains a (policy-question-driven) analysis of data pertaining to Wisconsin lakes, serving as a demonstration of reproducible data science practices. The project showcases how to transform Jupyter notebooks into multiple publication formats (HTML, PDF, and DOCX) while maintaining reproducibility and professional formatting. Also showcased are novel (off-label) uses of machine learning techniques for policy analysis.

## Reproducibility Features

### Quarto Integration

The project uses Quarto for document generation, supporting:
- Multiple output formats (HTML, PDF, DOCX)
- Automatic table of contents
- Bibliography (reference list) management

### Code Organization + Reproducibility Philosophy

- Analysis + manuscript both contained in a single Jupyter notebook)
- Figures include Quarto-supported labeling and cross-referencing

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
