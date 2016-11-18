run:
	python python/main.py `pwd`/data --summary-dir `pwd`/summary

test:
	python python/tests.py

clean:
	rm -rf `pwd`/data `pwd`/summary `pwd`/checkpoints/*
