.PHONY: run_builder run_inference install clean check runner_builder runner_inference
.DEFAULT_GOAL := runner_inference

run_builder: install
	cd src && poetry run python runner_builder.py

run_inference: install
	cd src && poetry run python runner_inference.py

install: pyproject.toml
	poetry install

clean:
	poetry run python -c "import pathlib, shutil, os, stat; fix = lambda f: (os.chmod(f, stat.S_IWRITE | stat.S_IREAD), os.unlink(f)); [(shutil.rmtree(p, onerror=lambda func, path, exc: (os.chmod(path, stat.S_IWRITE | stat.S_IREAD), func(path)))) for p in pathlib.Path('.').rglob('__pycache__') if p.exists()]"
	
check: 
	poetry run flake8 src/

runner_builder: check run_builder clean

runner_inference: check run_inference clean

