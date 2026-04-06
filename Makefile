.PHONY: install preprocess test lint run dashboard docker

install:
	pip install -e ".[dev]"

preprocess:
	python scripts/preprocess.py

test:
	pytest -v

lint:
	ruff check app/ tests/ scripts/

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/streamlit_app.py

docker:
	docker compose up --build
