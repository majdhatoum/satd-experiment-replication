.PHONY: image gold eval all clean

image:
	docker build -t satd-exp:latest .

gold:
	docker run --rm -v ${PWD}:/workspace -w /workspace satd-exp:latest \
		python analysis/build_gold_and_agreement.py

eval:
	docker run --rm -v ${PWD}:/workspace -w /workspace satd-exp:latest \
		python analysis/evaluate_and_plot.py

all: gold eval

clean:
	rm -f outputs/*.txt outputs/*.csv outputs/*.png
