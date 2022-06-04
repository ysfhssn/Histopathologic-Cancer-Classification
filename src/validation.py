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
  
  def predict(self):
    self.val_gen.reset()
    self.val_gen.batch_size = 1
    self.val_gen.shuffle = False

    self.probs = self.model.predict(self.val_gen, verbose=1)
    self.preds = (self.probs > 0.5) * 1
    return self.preds[:10]

  def updatePreds(self, threshold):
    if self.probs != None:
     self. preds = (self.probs > threshold) * 1

  def getAccuracyScore(self):
    if self.probs != None:
      return accuracy_score(self.val_gen.classes, self.preds)

  def displayConfusionMatrix(self, modelName):
    if self.val_gen != None and self.preds != []:
      cm = confusion_matrix(self.val_gen.classes, self.preds)
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
      
  def displayCurve(self):
    if self.val_gen != None and self.preds != None:
      fpr, tpr, thresholds = roc_curve(self.val_gen.classes,self. probs)
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
  