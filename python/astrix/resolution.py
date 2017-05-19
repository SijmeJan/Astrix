#!/usr/bin/python

import numpy as np
import os
from glob import glob
import argparse
import subprocess
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.tri as triang
import matplotlib.cm as cm
import readfiles
import parameterfile as pf

class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def CleanUp():
    for f in glob("*.vtk"):
        os.remove(f)
    for f in glob("*.dat"):
        os.remove(f)

def DensPlot2D(number, Px = 1.0, Py = 1.0):
    vert, tri, state = readfiles.readall('./', number, Px, Py)

    vertX = vert[:,0]
    vertY = vert[:,1]
    dens = state[:,2]

    plt.tricontourf(vertX, vertY, tri, dens, 100, cmap=cm.bwr)

parser = argparse.ArgumentParser()
parser.add_argument("directory")
args = parser.parse_args()
print(args.directory)

direc = os.path.abspath(args.directory)

with PdfPages('resolution.pdf') as pdf:
    res = [1, 2, 4, 8, 16]
    schemes = ['N']
    order = ['1', '2', '2']
    massMatrices = ['1']
    selectLumpFlags = ['0']

    error = np.zeros(len(res))
    h = 2.0/np.asarray(res)

    with cd(args.directory + 'run/scalar/advect/vortex/temp'):
        plt.figure(figsize=(7, 5))
        plt.rcParams.update({'axes.labelsize': 'large'})
        plt.rc('font', family='serif')
        plt.xscale('log')
        plt.yscale('log')

        for j in range(0, len(schemes)):
            pf.ChangeParameter('./astrix.in',
                               [['integrationScheme', schemes[j]]])
            pf.ChangeParameter('./astrix.in',
                               [['integrationOrder', order[j]]])
            #pf.ChangeParameter('./astrix.in',
            #                   [['massMatrix', massMatrices[j]]])
            #pf.ChangeParameter('./astrix.in',
            #                   [['selectiveLumpFlag', selectLumpFlags[j]]])

            for i in range(0, len(res)):
                n = res[i]
                #plt.figure(figsize=(5, 5))
                #plt.rcParams.update({'axes.labelsize': 'large'})
                #plt.rc('font', family='serif')
                #plt.gca().set_aspect('equal')
                #plt.xlim([-1,1])
                #plt.ylim([-1,1])

                #pf.ChangeParameter('./astrix.in',
                #                   [['equivalentPointsX', str(n)]])
                pf.ChangeParameter('./astrix.in',
                                   [['maxRefineFactor', str(n)]])
                subprocess.call([direc + "/bin/astrix", "-cl", "advect", "astrix.in"])
                #DensPlot2D(10, Px=2.0, Py=2.0)
                data = np.loadtxt('simulation.dat')
                data = np.transpose(data)
                t = data[0]
                er = data[1]
                m = data[2]

                #etot = ex + ey + et + ep

                #data = data[len(data)-1]
                error[i] = er[len(er) - 1]


                CleanUp()

                #plt.title('N = ' + str(n))
                #plt.xlabel('x')
                #plt.ylabel('y')
                #pdf.savefig()  # saves the current figure into a pdf page
                #plt.close()
                #plt.plot(t, etot - etot[0])
                #print(etot[99] - etot[0])

            #pdf.savefig()  # saves the current figure into a pdf page
            #plt.close()

            #print(error)
            #plt.figure(figsize=(5, 5))
            #plt.rcParams.update({'axes.labelsize': 'large'})
            #plt.rc('font', family='serif')
            #plt.xscale('log')
            #plt.yscale('log')
            plt.plot(h, error)

        plt.plot(h, 4.0*error[0]*h/h[0], linestyle='--')
        plt.plot(h, 0.5*error[0]*h*h/(h[0]*h[0]), linestyle='--')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
