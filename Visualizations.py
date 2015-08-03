from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure, show, hold
from scipy.stats import t
import numpy as np
import statsmodels.api as sm


def plotObservedData(indVar, depVar):
    fig = figure()
    ax = Axes3D(fig)
    x, y, z = [], [], []

    for i, j in zip(indVar, depVar):
        x.append(i[1])
        y.append(i[2])
        z.append(j)

    ax.scatter(x, y, z, c='blue', marker='o', s=1.8)
    ax.set_xlabel('Taxi Time')
    ax.set_ylabel('Day of Week')
    ax.set_zlabel('Extra Mins Flight Time')

    show()
    return
  
  
  def qqplotResiduals(zHat, indVar, depVar):
    residuals = depVar - zHat
    df = np.shape(zHat)[0]
    df -= len(indVar[0])
    
    probPlot = sm.ProbPlot(residuals, t, distargs=(df,))
    probPlot.qqplot()
    
    show()
    return


def plotZDifference(indVarTest, depVarTest, depVarHat):
    fig = figure()
    ax = fig.gca(projection='3d')
    hold(True)

    xSurf, ySurf = np.meshgrid(indVarTest[:,1], indVarTest[:,2])
    zTestDot, zHatSurf = np.meshgrid(depVarTest, depVarHat)

    ax.scatter(xSurf, ySurf, zHatSurf.T, c='green', marker='o', alpha=0.5)
    ax.scatter(xSurf, ySurf, zTestDot, c='yellow', marker='v')
    ax.set_xlabel('Taxi Time')
    ax.set_ylabel('Day of Week')
    ax.set_zlabel('Extra Mins Flight Time')

    show()
    return
