dir := src projects

style:
	black $(dir)
	isort $(dir)

tests:
	pytest -sv tests/

lint:
	flake8 $(dir)
