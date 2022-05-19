dir := src experiments

style:
	black $(dir)
	isort $(dir)

tests:
	pytest -sv tests/
