from homr.main_model1 import main_model1
from homr.main_model2 import main_model2, replace_extension
from homr.model import InputPredictions
from homr.debug import Debug
import pickle
import os
import glob

# allpath = glob.glob('images/testfiles/*.png')

# for i in range(len(allpath)):
# i   magePath = allpath[i]
imagePath = 'images/testfiles/1_tch_115.png'
print(imagePath)
fdebug = False

# load the saved pickle if it has gone through 1st model
picklePathPred = replace_extension(imagePath, '_pred.pkl')
picklePathDebug = replace_extension(imagePath, '_debug.pkl')
if not os.path.exists(picklePathPred) or not os.path.exists(picklePathDebug):
    print("no model found, running model1 prediction")
    predictions, debug = main_model1(imagePath=imagePath, finit=False, fdebug=fdebug, fcache=False)

    with open(picklePathPred, 'wb') as file:
        pickle.dump(predictions, file)

    with open(picklePathDebug, 'wb') as file:
        pickle.dump(debug, file)
else:
    print("loading existing model1 prediction")
    with open(picklePathPred, 'rb') as file:
        predictions = pickle.load(file)
    with open(picklePathDebug, 'rb') as file:
        debug = pickle.load(file)

# run the 2nd model
main_model2(predictions, debug, imagePath, fdebug=False, fcache=False)
