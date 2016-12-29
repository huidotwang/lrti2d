# VTI/TTI wavefield propagators

### INSTALL
Make sure you have installed FFTW3 and set up environment variables `FFTW_INC` and `FFTW_LIB`. To check, 

```bash
echo $FFTW_INC
echo $FFTW_LIB
```

Go to the root directory and compile the source codes:

```bash
$ scons
```

### TESTING
```bash
cd demos
scons
```

You should be able to see wave propagations in constant VTI and TTI media as well as passing the dot-product tests (all four modes).

