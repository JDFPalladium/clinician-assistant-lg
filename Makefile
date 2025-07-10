install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint: 
	pylint --disable=R,C app.py chatlib

test:
	PYTHONPATH=. pytest -vv 

format:
	black app.py chatlib

run:
	python app.py