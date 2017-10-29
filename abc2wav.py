import sys,os
from subprocess import call

call(['abc2midi' ,sys.argv[1] ,'-o', sys.argv[1].split('.')[0]+'.mid'])
call(['timidity', sys.argv[1].split('.')[0]+'.mid', '-Ow'])
os.remove(sys.argv[1].split('.')[0]+'.mid')