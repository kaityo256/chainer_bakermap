all: a.out

a.out: import.cpp
	g++ -std=c++11 $< -o $@

clean:
	rm -f a.out
