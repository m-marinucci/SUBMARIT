# How to View SUBMARIT Documentation

## Option 1: Build and View HTML Documentation (Recommended)

### Build the documentation:
```bash
# Activate your virtual environment
source venv/bin/activate

# Navigate to docs directory
cd docs

# Build HTML documentation
make html

# Or if make is not available:
sphinx-build -b html source build/html
```

### View in browser:
```bash
# Open the main page in your default browser (macOS)
open build/html/index.html

# Or navigate to the file directly:
# file:///Users/numinate/PY/SubmarketIdentificationTesting/docs/build/html/index.html
```

## Option 2: Use sphinx-autobuild for Live Preview

```bash
# Install sphinx-autobuild
pip install sphinx-autobuild

# Run auto-building server
sphinx-autobuild source build/html

# Opens at http://127.0.0.1:8000
# Auto-refreshes when you edit RST files
```

## Option 3: View Raw RST Files

### In VS Code:
- Install "reStructuredText" extension by LeXtudio
- Open any .rst file for syntax highlighting
- Use Ctrl+Shift+V (Cmd+Shift+V on Mac) for preview

### In PyCharm:
- Built-in RST support with preview
- Open .rst file and click preview tab

### Online RST Viewers:
- https://livesphinx.herokuapp.com/
- http://rst.ninjs.org/

## Option 4: Simple HTTP Server

```bash
# After building HTML docs
cd docs/build/html
python3 -m http.server 8000

# View at http://localhost:8000
```

## Browser Compatibility

Any modern browser works perfectly:
- **Chrome** - Recommended
- **Firefox** - Excellent support
- **Safari** - Native on macOS
- **Edge** - Good support

## Quick Start

```bash
# Complete command sequence:
cd /Users/numinate/PY/SubmarketIdentificationTesting
source venv/bin/activate
cd docs
make html
open build/html/index.html
```

## Troubleshooting

If `make html` fails:
```bash
# Install Sphinx if needed
pip install sphinx sphinx-rtd-theme nbsphinx

# Try direct build
sphinx-build -b html source build/html
```

## Documentation Structure

Once built, you'll find:
- `index.html` - Main documentation page
- `installation.html` - Installation guide
- `quickstart.html` - Getting started tutorial
- `api.html` - Complete API reference
- `migration_guide.html` - MATLAB to Python guide
- And more...

The HTML documentation includes:
- Navigation sidebar
- Search functionality
- Code syntax highlighting
- Cross-references
- Mobile-responsive design