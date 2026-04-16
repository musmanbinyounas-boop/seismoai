# SeismoAI

A Python package for seismic data analysis and visualization.

## Features

- Seismic data I/O operations (`seismoai_io`)
- Seismic data visualization (`seismoai_viz`)

## Installation

### From source
```bash
git clone <repository-url>
cd seismoai
pip install -e .
```

### Development setup
```bash
pip install -e ".[dev]"
```

## Usage

```python
import seismoai_io
import seismoai_viz

# Load seismic data
data = seismoai_io.load_data("data/27_...sgy")

# Visualize
seismoai_viz.plot_seismic(data)
```

## Project Structure

```
seismoai/
├── data/                 # Seismic data files (.sgy)
├── seismoai_io/          # I/O operations package
├── seismoai_viz/         # Visualization package
├── pyproject.toml        # Project configuration
├── README.md            # This file
├── requirements.txt     # Additional requirements
└── .gitignore          # Git ignore rules
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

[Add license information here]