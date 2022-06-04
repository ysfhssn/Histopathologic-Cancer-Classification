from models import model1, model2
from datasets import histopathologic_cancer_detection
import validation

BASE_PATH = '/Users/diez/Documents/Histopathologic-Cancer-Detection/src/dataset/histopathologic-cancer-detection/'
TRAIN_DIR_PATH = f'{BASE_PATH}train/'
TEST_DIR_PATH = f'{BASE_PATH}test/'
BATCH_SIZE = 32
EPOCHS = 1
SAMPLES_SIZE = 10_000

def setupDataset():
  dataset = histopathologic_cancer_detection.dataset(BASE_PATH)
  """ dataset.displaySamples(TRAIN_DIR_PATH) """

  dataset.balanceDataset()

  dataset.initiateDataTensor(TRAIN_DIR_PATH, BATCH_SIZE, SAMPLES_SIZE)
  return dataset
 
def TrainTestModel(modelName, dataset):
  model = None
  if modelName == "Model1":
    model = model1.model1()
  else: 
    model = model2.model2()
  model.fit(dataset.train_gen, dataset.val_gen, EPOCHS)
  model.summary()

  test = validation.validation(dataset.val_gen, model.instance)
  print(test.predict())
  print(test.getAccuracyScore())
  
  test.updatePreds(test.displayCurve())
  
  test.displayConfusionMatrix(modelName)
  
def main():
  dataset = setupDataset()
  TrainTestModel("Model1", dataset)
  TrainTestModel("Model2", dataset)#

main()