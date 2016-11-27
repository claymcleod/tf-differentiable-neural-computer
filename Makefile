run:
	@mkdir -p `pwd`/checkpoints
	python python/main.py copy `pwd`/data --summary-dir `pwd`/summary

test:
	python python/tests.py

clean:
	rm -rf `pwd`/data `pwd`/summary `pwd`/checkpoints/*
