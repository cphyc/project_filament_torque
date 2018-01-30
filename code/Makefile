FC=gfortran
F2PY=f2py

all: correlations.o

%.o: %.f90
	$(FC) -fPIC -c -o $@ $^


py: correlations.f90
	f2py -m correlations $^ -c

clean:
	rm -rf 	*f2pywrappers* *.o *.mod
