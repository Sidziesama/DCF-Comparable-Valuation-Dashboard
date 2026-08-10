# Notebook 01 — Data Exploration

Use this notebook to inspect:
- SEC historical statements
- missing XBRL tags
- fiscal year coverage
- market data
- data validation

Suggested first cells:

```python
import sys
sys.path.append("../src")

from sec_data import build_historical

hist = build_historical("0001403161", 5)
display(hist)
```
