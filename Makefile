dir := src experiments

style:
	black $(dir)
	isort $(dir)
