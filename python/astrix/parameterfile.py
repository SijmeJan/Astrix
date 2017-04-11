#!/usr/bin/python

import os

def ChangeParameter(inFileName, parameter):
    """Edit a valid Astrix input file to change parameters

    Given a valid Astrix input file inFileName, edit it to change the parameters as listed in parameter

    :param inFileName: Valid Astrix input parameter file.
    :param parameter: List of pairs of strings [parameterName, parameterValue]

    :type inFileName: string
    :type parameter: List of string pairs
    """
    fullPath = os.path.abspath(inFileName)
    direc = fullPath.rsplit("/",1)
    outFileName = direc[0] + '/temp.in'

    # Open input file
    inFile = open(inFileName, "r")
    # Open input file
    outFile = open(outFileName, "w")

    # Process file line by line
    for line in inFile:
        # By default, just copy line
        lineNew = line

        for p in parameter:
            if(p[0] in line):
                s = list(line)
                v = list(p[1])

                foundSpace = False
                written = False
                j = -1
                for i in range(0, len(s)):
                    if(s[i] == ' ' or s[i] == '\t'):
                        foundSpace = True
                    if(s[i] != ' ' and s[i] != '\t' and
                    foundSpace == True and written == False):
                        j = i
                        foundSpace = False
                    if(j >= 0):
                        written = True
                        if(i - j < len(v)):
                            s[i] = v[i - j]
                        if(i - j >= len(v) and foundSpace == False):
                            s[i] = ' '


                # Join up line from characters
                lineNew = "".join(s)

        # Write line to output file
        outFile.write(lineNew)

    # Close all files
    inFile.close()
    outFile.close()

    # Replace old input file with new
    os.remove(inFileName)
    os.rename(outFileName, inFileName)
