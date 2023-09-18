import Dump
import fileForces

import numpy as np
import os
import pandas as pd
import json

AgEpsilon: float
AgSigma: float
AlEpsilon: float
AlSigma: float
AlEps: float
AlAlpha: float

def Radius (Positions, i, j): #подсчет модуля радиус-вектора между двумя атомами
    return np.sqrt(np.sum(np.square(Positions[i]-Positions[j])))

def LennardJones (Positions, Forces, Velocities): #подсчет потенциала Леннарда-Джонса в момент времени
    atomsNum, dim = Velocities.shape

    for i in range (1, atomsNum):
        radius = Radius(Positions, 0, i)
        force = 48 * AgEpsilon * ((AgSigma ** 12) / (radius ** 7) - (AgSigma ** 6) / (radius ** 7))

        Forces[0][0] += -(Positions[0][0] - Positions[i][0]) * force / radius
        Forces[0][1] += -(Positions[0][1] - Positions[i][1]) * force / radius
        Forces[0][2] += -(Positions[0][2] - Positions[i][2]) * force / radius

def Morze (Positions, Forces, Velocities): #подсчет потенциала Морзе в момент времени
    atomsNum, dim = Velocities.shape

    for i in range(1, atomsNum):
        for j in range(i + 1, atomsNum):
            radius = Radius(Positions, i, j)
            force = AlEps * np.exp(-2 * AlAlpha * radius) * (AlAlpha * np.exp(AlAlpha * radius)-4 * AlAlpha)

            Forces[i][0] += -(Positions[i][0] - Positions[j][0]) * force / radius
            Forces[i][1] += -(Positions[i][1] - Positions[j][1]) * force / radius
            Forces[i][2] += -(Positions[i][2] - Positions[j][2]) * force / radius

            Forces[j][0] -= -(Positions[i][0] - Positions[j][0]) * force / radius
            Forces[j][1] -= -(Positions[i][1] - Positions[j][1]) * force / radius
            Forces[j][2] -= -(Positions[i][2] - Positions[j][2]) * force / radius

def Acceleration (Masses, Forces, Velocities): #подсчет ускорения в момент времени (зная значение потенциалов и сил)
    atomsNum, dim = Velocities.shape
    acceleration = np.array([[0.0 for i in range(dim)] for j in range(atomsNum)])

    for  i in range (0, atomsNum):
        acceleration[i][0] += Forces[i][0] / Masses[i]
        acceleration[i][1] += Forces[i][1] / Masses[i]
        acceleration[i][2] += Forces[i][2] / Masses[i]

    return acceleration

def Verle (Accelerations, Positions, Velocities, prevPositions, dt): #вычисление новых позиций атомов алгоритмом Верле

    atomsNum, dim = Velocities.shape
    newPositions = np.array([[0.0 for i in range(dim)] for j in range(atomsNum)])

    for i in range (0, atomsNum):
        newPositions[i][0] += 2 * Positions[i][0] - prevPositions[i][0] + Accelerations[i][0] * (dt ** 2)
        newPositions[i][1] += 2 * Positions[i][1] - prevPositions[i][1] + Accelerations[i][1] * (dt ** 2)
        newPositions[i][2] += 2 * Positions[i][2] - prevPositions[i][2] + Accelerations[i][2] * (dt ** 2)

    return newPositions

#поиск атомов, ближайших к падающему на решетку атому Ag.
#Мы будем рассматривать силы взаимодействия между этими атомами и атомом Ag.
def findNearestAtoms(Positions, Velocities):
    atomsNum, dim = Velocities.shape
    nearestAtoms = np.array([[0, 0] for i in range(atomsNum-1)])

    for i in range (1, atomsNum):
        nearestAtoms[i-1][0]+=i
        nearestAtoms[i - 1][1] += Radius(Positions, 0, i)

    nearest = sorted(nearestAtoms, reverse=True, key=lambda x: x[1])
    res = np.array([0 for i in range (5)])
    for i in range (5):
        res[i] += nearest[i][0]
    return res


def getData ():
    parameters = {
        'AgRadius': 0,
        'AgMass': 0,
        'AgEpsilon': 0,
        'AgSigma': 0,

        'AlNumOfAtoms': 0,
        'AlRadius': 0,
        'AlMass': 0,
        'AlEpsilon': 0,
        'AlSigma': 0,
        'AlEps': 0,
        'AlAlpha': 0,

        'TimeStep': 1e-16,
        'Steps': 100,
        'OutputFrequency': 2,
        'Borders': [[], [], []],
        'OutputFileName': 'output.dump'
    }

    dataObj = pd.read_csv('Data.csv', delimiter=';')
    csvList = [tuple(row) for row in dataObj.values]
    for x in csvList:
        if x[0] == 'Borders':
            parameters[x[0]] = json.loads(x[1])
        elif x[0] == 'OutputFileName':
            parameters[x[0]] = x[1]
        else:
            parameters[x[0]] = float(x[1])
    parameters['AlNumOfAtoms'] = int(parameters['AlNumOfAtoms'])
    parameters['Steps'] = int(parameters['Steps'])
    parameters['OutputFrequency'] = int(parameters['OutputFrequency'])

    return parameters

def start():
    global AgEpsilon, AgSigma, AlEpsilon, AlSigma, AlEps, AlAlpha
    parameters = getData()

    #Считывание данных из кортежа
    AgEpsilon, AgSigma, AlEpsilon, AlSigma, AlEps, AlAlpha = \
        parameters['AgEpsilon'], parameters['AgSigma'], parameters['AlEpsilon'], \
        parameters['AlSigma'], parameters['AlEps'], parameters['AlAlpha']

    AgRadius, AgMass, AlNumOfAtoms, AlRadius, AlMass = \
        parameters['AgRadius'], parameters['AgMass'], parameters['AlNumOfAtoms'], \
        parameters['AlRadius'], parameters['AlMass']

    TimeStep, Steps, OutputFrequency, Borders, OutputFileName = \
        parameters['TimeStep'], parameters['Steps'], parameters['OutputFrequency'], \
        parameters['Borders'], parameters['OutputFileName']

    dim = len(Borders)  # Вычислние размерности

    #Преобразование значений в массивы для удобства вычислений

    AgPosition = np.array([0.0 for i in range(dim)])
    AgVelocity = np.array([0.0 for i in range(dim)])
    AgForce = np.array([0.0 for i in range(dim)])
    AgVelocity = np.array([0.0, 0.0, 0.0])
    AgForce = np.array([0.0, 0.0, 0.0])

    AlMass = np.ones(AlNumOfAtoms) * AlMass
    AlRadius = np.ones(AlNumOfAtoms) * AlRadius
    AlVelocity = np.array([[0.0 for i in range(dim)] for j in range(AlNumOfAtoms)])
    AlForce = np.array([[0.0 for i in range(dim)] for j in range(AlNumOfAtoms)])
    AlPositions = np.array([[0.0 for i in range(dim)] for j in range(AlNumOfAtoms)])

    # Построение исходного состояния системы
    AgPosition = np.array([1 / 4, 1 / 4, 3 / 2])

    if AlNumOfAtoms % 4 == 0:
        coeff = 1 / (AlNumOfAtoms // 4)
    elif AlNumOfAtoms % 3 == 0:
        coeff = 1 / ((AlNumOfAtoms + 3) // 4)
    elif AlNumOfAtoms % 2 == 0:
        coeff = 1 / ((AlNumOfAtoms + 2) // 4)
    else:
        coeff = 1 / ((AlNumOfAtoms + 1) // 4)

    for i in range(0, AlNumOfAtoms):
        AlPositions[i][0], AlPositions[i][1] = 1 / 4 + i % 4 * 1 / 4, i // 4 * coeff + coeff

    for i in range(dim):  # Подгонка значений относительно границ области
        AlPositions[:, i] = Borders[i][0] + (Borders[i][1] - Borders[i][0]) * AlPositions[:, i]
        AgPosition[i] = Borders[i][0] + (Borders[i][1] - Borders[i][0]) * AgPosition[i]

    Positions = np.append([AgPosition], AlPositions, axis=0)
    Velocities = np.append([AgVelocity], AlVelocity, axis=0)
    Masses = np.append([AgMass], AlMass)
    Radius = np.append([AgRadius], AlRadius)
    Forces = np.append([AgForce], AlForce, axis=0)

    if os.path.exists(OutputFileName):
        os.remove(OutputFileName)

    nearestAtoms = findNearestAtoms(Positions, Velocities)
    nearestAtoms.sort()

    nearestForces = np.array([[[0.0 for i in range(dim)]for j in range (len(nearestAtoms))]for k in range (Steps+1)])

    step = 0
    prevPositions = np.array([[0.0 for i in range(dim)] for j in range(len(Velocities))])
    for i in range (len(prevPositions)):
        prevPositions[i] = np.copy(Positions[i])

    # Для каждой итерации:
    while step < Steps:
        step += 1

        # Подсчет потенциалов и вычисление сил
        LennardJones(Positions, Forces, Velocities)
        for i in range (len(nearestAtoms)):
            nearestForces[step-1][i] += Forces[nearestAtoms[i]]

        # Запись в файл данных о взаимодействии атома Ag c ближайшими атомами металла
        fileForces.writeOutput(nearestForces, nearestAtoms, step, Positions)

        # Подсчет потенциалов и вычисление сил. Вычисление ускорений.
        Morze(Positions, Forces, Velocities)
        Accelerations = Acceleration(Masses, Forces, Velocities)

        # Подсчет новых координат атомов
        if (step == 1):
            for i in range (len(Positions)):
                Positions[i][0] = prevPositions[i][0] + 1/2 * Accelerations[i][0] * np.square(TimeStep)
                Positions[i][1] = prevPositions[i][1] + 1 / 2 * Accelerations[i][1] * np.square(TimeStep)
                Positions[i][2] = prevPositions[i][2] + 1 / 2 * Accelerations[i][2] * np.square(TimeStep)
        else:
            newPositions = Verle(Accelerations, Positions, Velocities, prevPositions, TimeStep)
            prevPositions = np.copy(Positions)
            Positions = np.copy(newPositions)

        Velocities += Accelerations * TimeStep

        Dump.writeOutput(OutputFileName, AlNumOfAtoms + 1, step, Borders,
                         radius=Radius, pos=Positions, v=Velocities)