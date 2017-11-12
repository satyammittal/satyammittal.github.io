from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from sklearn.externals import joblib

from word_vectors import *

alldata = ClassificationDataSet(len(word_vectors[0][1]), 1, nb_classes=2)
for i, tup in enumerate(word_vectors):
    alldata.addSample(tup[1], tup[0])

tstdata, trndata = alldata.splitWithProportion(0.25)

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
trainer.trainUntilConvergence(maxEpochs = 30)

joblib.dump(fnn, "fnn_model.pkl")


