from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def create_roc_curve(labels, scores, positive_label)
  fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=positive_label)
  roc_auc = auc(fpr, tpr)

  plt.title('Receiver Operating Characteristic')
  plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
  plt.legend(loc='lower right')
  plt.plot([0,1],[0,1],'r--')
  plt.xlim([-0.1,1.2])
  plt.ylim([-0.1,1.2])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()

y = np.array([0, 0, 1, 1])
scores = np.array([0.1, 0.4, 0.35, 0.8])

create_roc_curve(y, scores, 1)
