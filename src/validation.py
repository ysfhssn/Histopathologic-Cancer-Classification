from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import accuracy_score
class validation:
  model = None
  val_gen = None
  preds = None
  probs = None
  
  def __init__(self, val_gen, model) -> None:
    self.model = model
    self.val_gen = val_gen
  
  def predict():
    val_gen.reset()
    val_gen.batch_size = 1
    val_gen.shuffle = False

    probs = model.predict(val_gen, verbose=1)
    preds = (probs > 0.5) * 1
    return preds[:10]

  def updatePreds(threshold):
    if probs != None:
      preds = (probs > threshold) * 1

  def getAccuracyScore():
    return accuracy_score(val_gen.classes, preds)

  def displayConfusionMatrix(modelName):
    if val_gen != None and preds != []:
      cm = confusion_matrix(val_gen.classes, preds)
      print(cm)
      ax = sns.heatmap(cm/np.sum(cm), annot=True, 
                  fmt='.2%', cmap='Blues')

      ax.set_title(modelName + ' validation confusion matrix\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('\nActual Values ')

      ## Ticket labels - List must be in alphabetical order
      ax.xaxis.set_ticklabels(['False','True'])
      ax.yaxis.set_ticklabels(['False','True'])

      ## Display the visualization of the Confusion Matrix.
      plt.show()
      
  def displayCurve():
    if val_gen != None and preds != None:
      fpr, tpr, thresholds = roc_curve(val_gen.classes, probs)
      auc_m1 = auc(fpr, tpr)

      plt.figure(figsize=(5, 5), dpi=100)
      plt.plot(fpr, tpr, linestyle='-', label='auc = %0.3f' % auc_m1)
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.legend()
      plt.show()

      print(fpr)
      print(tpr)
      print(thresholds)
      print("BEST THRESHOLD:", thresholds[np.argmax(tpr - fpr)])
      return thresholds[np.argmax(tpr - fpr)]
  