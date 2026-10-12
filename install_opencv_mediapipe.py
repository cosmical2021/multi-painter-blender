import subprocess
import sys
import os

#Path to python.exe
python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
py_lib = os.path.join(sys.prefix, 'lib', 'site-packages','pip')

#install opencv
subprocess.call([python_exe, py_lib, "install", "opencv_python"])
#install mediapipe
subprocess.call([python_exe, py_lib, "install", "mediapipe"])