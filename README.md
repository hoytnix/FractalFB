# FractalFB - Facebook Ad Analysis and Reporting Tool

## Overview

FractalFB is a powerful Python-based tool for analyzing Facebook ad performance data and generating comprehensive PDF reports with visualizations and actionable insights. It processes data from Facebook Ads Manager and evaluates performance against user-defined KPI ranges.

## Features

- Automated PDF report generation with detailed ad performance analysis
- Interactive funnel visualizations and performance graphs
- Historical performance tracking and trend analysis
- Customizable KPI ranges and performance metrics
- Automated recommendations for ad optimization
- Sphinx documentation generation

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.x
- pandas
- PyYAML
- FPDF
- Pillow
- Sphinx (for documentation)

## Usage

### Basic Usage

```bash
python app.py --csv facebook_ads.csv --yaml settings.yaml --output report.pdf
```

### Generate Documentation

```bash
python app.py --generate-docs --docs-dir docs
```

### Command Line Arguments

- `--csv`: Path to Facebook ads CSV file (default: facebook_ads.csv)
- `--yaml`: Path to settings YAML file (default: settings.yaml)
- `--output`: Output PDF file path (default: kpi_report.pdf)
- `--database`: Path to YAML database file (default: reports_database.yaml)
- `--generate-docs`: Generate Sphinx documentation
- `--docs-dir`: Directory for Sphinx documentation (default: docs)

## Configuration

Create a `settings.yaml` file with your organization name and KPI ranges:

```yaml
organization: "Your Organization Name"
kpi_ranges:
  "Quality ranking":
    min: 7
    max: 10
  "CTR (link click-through rate)":
    min: 0.5
    max: 3
  "Cost per results":
    min: 0.01
    max: 30
```

## Output

The tool generates:

1. Comprehensive PDF reports with:
   - KPI analysis
   - Funnel visualizations
   - Performance graphs
   - Recommendations
2. Historical performance database
3. Sphinx documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Copyright

Copyright © Michael Hoyt 2024. All Rights Reserved.

## Support

For support, please open an issue in the GitHub repository.

## Authors

- Michael Hoyt

## Acknowledgments

Special thanks to all contributors and users of FractalFB.
