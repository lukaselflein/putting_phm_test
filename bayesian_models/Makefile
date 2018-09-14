all: calc convert plot clean

calc:
	python3 simple_queue.py

convert:
	python3 convert_trace_to_csv.py

plot:
	python3 plot_csv.py
clean:
	rm -f *.pyc
	rm -rf __pycache__
