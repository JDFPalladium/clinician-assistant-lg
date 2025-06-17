install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint: 
	pylint --disable=R,C 

test:
	PYTHONPATH=. pytest -vv 

format:
	black 