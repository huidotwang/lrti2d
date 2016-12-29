import os

platform = ARGUMENTS.get('OS', Platform())
# execute with "scons -Q OS=linux"

incdir = "#build/inc"
libdir = "#build/lib"
bindir = "#build/bin"

cflags = (
          " -O3 -std=gnu99 -fPIC -DHAVE_INLINE "
          " -Wall -Wextra "
          " -fopenmp -DSF_HAS_FFTW -DSF_HAS_FFTW_OMP ")

cxxflags = (
          " -O3 -fPIC -DHAVE_INLINE "
          " -Wall -Wextra "
          " -fopenmp -DSF_HAS_FFTW -DSF_HAS_FFTW_OMP ")

AddOption('--prefix',
          dest='prefix',
          type='string',
          nargs=1,
          action='store',
          metavar='DIR',
          help='installation prefix')

compiler = dict()
compiler['cc'] = os.environ['CC']
compiler['cxx'] = os.environ['CXX']
compiler['mpicc'] = os.environ['MPICC']
compiler['mpicxx'] = os.environ['MPICXX']

libpath = ['.', '#/common']
cpppath = ['.', '#/inc']

rsfroot = os.environ['RSFROOT']
rsf_lib = rsfroot+'/lib'
rsf_inc = rsfroot+'/include'

fftw_inc = os.environ['FFTW_INC']
fftw_lib = os.environ['FFTW_LIB']

cpppath.append(rsf_inc)
libpath.append(rsf_lib)
libs = ['rsf','su','m','gomp']

env = Environment(PREFIX = GetOption('prefix'),
                  PLATFORM = platform,
                  BINDIR = bindir,
                  INCDIR = incdir,
                  LIBDIR = libdir,
                  CPPPATH = cpppath,
                  LIBPATH = libpath,
                  CC = compiler['cc'],
                  CXX = compiler['cxx'], 
                  CFLAGS = cflags,
                  CXXFLAGS = cxxflags)
env['SHLIBPREFIX'] = ''

install_bins = []
install_libs = []

install_libs += SConscript('common/SConscript', exports=['env'])

mexport = ['env', 'compiler', 'libs', 'fftw_inc', 'fftw_lib']


eigen_inc = "#/deps/Eigen"
install_bins += SConscript('tti/SConscript', exports=mexport+['eigen_inc'])

env.Alias("install_lib", env.Install('$PREFIX/lib', install_libs))
env.Alias("install_bin", env.Install('$PREFIX/bin', install_bins))
env.Alias('install',['install_lib', 'install_bin'])
