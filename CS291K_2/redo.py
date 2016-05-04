import os
import sys

path = sys.argv[1]

def redo(path):
    os.system('python conv_net.py %s' % path)


if __name__ == "__main__":
    redo(path)
