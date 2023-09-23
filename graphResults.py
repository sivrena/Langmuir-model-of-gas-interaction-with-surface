import numpy as np
import matplotlib.pyplot as plt

def Radius (Positions, i, j): #подсчет модуля радиус-вектора между двумя атомами
    return np.sqrt(np.sum(np.square(Positions[i]-Positions[j])))

def graphForces (forces, atomNumber, stepsNumber):
    axisX = np.array([0 for i in range (stepsNumber)])
    axisY = np.array([0 for i in range(stepsNumber)])
    axisZ = np.array([0 for i in range(stepsNumber)])

    for i in range (stepsNumber):
        axisX[i] += i
        axisY[i] += forces[i][atomNumber][0] + forces[i][atomNumber][1]

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    plt.plot(axisX, axisY)
    plt.savefig('graphForces.jpg', bbox_inches='tight')
    plt.show()

def graphLennardJones (Positions, atomNumber, AgEpsilon, AgSigma, step, X, Y):
    radius = Radius(Positions, 0, atomNumber)
    force = -48 * AgEpsilon * ((AgSigma ** 12) / (radius ** 13) - (AgSigma ** 6) / (radius ** 7))

    X[step-1] += radius
    Y[step-1] += force

def graphMorze(Positions, i, j, AlEps, AlAlpha, step, X, Y):
    radius = Radius(Positions, i, j)
    force = -(AlEps * np.exp(-2 * AlAlpha * radius) * (AlAlpha * np.exp(AlAlpha * radius) - 4 * AlAlpha))

    X[step - 1] += radius
    Y[step - 1] += force