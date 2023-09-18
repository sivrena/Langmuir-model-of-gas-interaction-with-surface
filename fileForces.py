import numpy as np

def writeOutput(forces, atoms, step, Positions):

    fw = open ("nearestAtomsForces.txt", "a")
    fw.write("\nStep " + str(step) + "\n")

    for i in range (len(atoms)):
        radius = np.sqrt(np.sum(np.square(Positions[0]-Positions[atoms[i]])))

        fw.write("Atom # " + str(atoms[i]) + ", coordinates " + str(Positions[atoms[i]]) +\
                     ", distance to Ag " + str(radius) + "\n")
        fw.write("Interaction force " + str(forces[step-1][i]) +"\n\n")

    if (step==201):
        fw.close()