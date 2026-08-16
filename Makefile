PYTHON ?= python3

.PHONY: setup test visa microsoft demo dashboard clean-demo

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

visa:
	$(PYTHON) src/pipeline.py --config config/company.yaml

microsoft:
	$(PYTHON) src/pipeline.py --config config/microsoft.yaml

demo:
	$(PYTHON) src/portfolio_demo.py

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py -- --config config/company.yaml
