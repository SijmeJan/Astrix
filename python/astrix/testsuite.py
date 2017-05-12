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

def DensPlot1D(number, lbl=''):
    vert, tri, state = readfiles.readall('./', number)
    nVertex = (len(vert) - 4)/3 + 1

    vert = np.transpose(vert)
    x = vert[0,0:nVertex]

    state = np.transpose(state)
    dens = state[0,0:nVertex]

    return plt.plot(x, dens, label = lbl)

def DensPlot2D(number, Px = 1.0, Py = 1.0):
    vert, tri, state = readfiles.readall('./', number, Px, Py)

    vertX = vert[:,0]
    vertY = vert[:,1]
    dens = state[:,0]

    plt.tricontourf(vertX, vertY, tri, dens, cmap=cm.bwr)

def TriPlot(number, Px = 1.0, Py = 1.0):
    vert, tri, state = readfiles.readall('./', number, Px, Py)

    vertX = vert[:,0]
    vertY = vert[:,1]

    plt.triplot(vertX, vertY, tri, lw=1, color='k')

parser = argparse.ArgumentParser()
parser.add_argument("directory")
args = parser.parse_args()
print(args.directory)

direc = os.path.abspath(args.directory)

with PdfPages('testsuite.pdf') as pdf:
    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'axes.labelsize': 'large'})
    plt.rc('font', family='serif')

    # Sod shock tube test
    with cd(args.directory + 'run/euler/sod'):
        pf.ChangeParameter('./astrix.in', [['integrationScheme', 'B'],
                                           ['integrationOrder', '2']])
        subprocess.call([direc + "/bin/astrix", "astrix.in"])
        B2, = DensPlot1D(20, 'B2')
        CleanUp()
    with cd(args.directory + 'run/euler/sod'):
        pf.ChangeParameter('./astrix.in', [['integrationScheme', 'N'],
                                           ['integrationOrder', '1']])
        subprocess.call([direc + "/bin/astrix", "astrix.in"])
        N1, = DensPlot1D(20, 'N1')
        CleanUp()

    plt.legend(handles=[B2, N1], loc=0)
    plt.title('Sod shock tube')
    plt.xlabel('x')
    plt.ylabel(r'$\rho$')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Interacting blast waves
    with cd(args.directory + 'run/euler/blast'):
        pf.ChangeParameter('./astrix.in', [['integrationScheme', 'BX'],
                                           ['integrationOrder', '2']])
        subprocess.call([direc + "/bin/astrix", "astrix.in"])
        Bx2, = DensPlot1D(38, 'Bx2')
        CleanUp()
    with cd(args.directory + 'run/euler/blast'):
        pf.ChangeParameter('./astrix.in', [['integrationScheme', 'N'],
                                           ['integrationOrder', '1']])
        subprocess.call([direc + "/bin/astrix", "astrix.in"])
        N1, = DensPlot1D(38, 'N1')
        CleanUp()

    plt.legend(handles=[Bx2, N1], loc=0)
    plt.title('Interacting blast waves')
    plt.xlabel('x')
    plt.ylabel(r'$\rho$')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Linear sound wave
    with cd(args.directory + 'run/euler/linear'):
        pf.ChangeParameter('./astrix.in', [['integrationScheme', 'N'],
                                           ['integrationOrder', '1']])
        subprocess.call([direc + "/bin/astrix", "astrix.in"])
        N1, = DensPlot1D(10, 'N1')
        CleanUp()
    with cd(args.directory + 'run/euler/linear'):
        pf.ChangeParameter('./astrix.in', [['integrationScheme', 'LDA'],
                                           ['integrationOrder', '2']])
        subprocess.call([direc + "/bin/astrix", "astrix.in"])
        LDA2, = DensPlot1D(10, 'LDA2')
        exact, = DensPlot1D(0, 'exact')
        CleanUp()

    plt.legend(handles=[LDA2, N1, exact], loc=0)
    plt.title('Linear sound wave')
    plt.xlabel('x')
    plt.ylabel(r'$\rho$')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Flow past cylinder
    plt.gca().set_aspect('equal')
    with cd(args.directory + 'run/euler/cyl/'):
        subprocess.call([direc + "/bin/astrix", "astrix.in"])
        DensPlot2D(20, Px = 2.0, Py = 2.0)
        TriPlot(20, Px = 2.0, Py = 2.0)
        CleanUp()

    plt.title('Flow past cylinder')
    plt.xlabel('x')
    plt.ylabel('y')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Noh problem
    plt.gca().set_aspect('equal')
    with cd(args.directory + 'run/euler/noh/'):
        subprocess.call([direc + "/bin/astrix", "astrix.in"])
        DensPlot2D(20)
        TriPlot(20)
        CleanUp()

    plt.title('Noh problem')
    plt.xlabel('x')
    plt.ylabel('y')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # 2D Riemann
    plt.gca().set_aspect('equal')
    with cd(args.directory + 'run/euler/riemann/'):
        subprocess.call([direc + "/bin/astrix", "astrix.in"])
        DensPlot2D(8)
        TriPlot(8)
        CleanUp()

    plt.title('2D Riemann')
    plt.xlabel('x')
    plt.ylabel('y')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Vortex advection
    plt.gca().set_aspect('equal')
    with cd(args.directory + 'run/euler/vortex/'):
        subprocess.call([direc + "/bin/astrix", "astrix.in"])
        DensPlot2D(10)
        CleanUp()

    plt.title('Vortex advection')
    plt.xlabel('x')
    plt.ylabel('y')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Kelvin-Helmholtz
    plt.gca().set_aspect('equal')
    with cd(args.directory + 'run/euler/kh/'):
        subprocess.call([direc + "/bin/astrix", "astrix.in"])
        DensPlot2D(35)
        CleanUp()

    plt.title('Kelvin-Helmholtz')
    plt.xlabel('x')
    plt.ylabel('y')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Rayleigh-Taylor
    #plt.gca().set_aspect('equal')
    #with cd(args.directory + 'run/euler/source/'):
    #    subprocess.call([direc + "/bin/astrix", "astrix.in"])
    #    DensPlot2D(75, Px = 2.0, Py = 2.0)
    #    TriPlot(75, Px = 2.0, Py = 2.0)
    #    CleanUp()

    #plt.title('Rayleigh-Taylor')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #pdf.savefig()  # saves the current figure into a pdf page
    #plt.close()
