from nolearn.lasagne.visualize import draw_to_notebook
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from nolearn.lasagne.visualize import plot_saliency
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm

#This prints graph of train and valid loss
train_loss = np.array([i["train_loss"] for i in net0.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net0.train_history_])
pyplot.plot(train_loss, linewidth=3, label="train")
pyplot.plot(valid_loss, linewidth=3, label="valid")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.show()

#saves architexture to notebook -- how cool!!
draw_to_notebook(net0)

#we need to define a 
results = net0.predict(X) #This NEEDS to be the x that we have
cm = confusion_matrix(y, results)
pyplot.matshow(cm)
pyplot.title('Confusion matrix')
pyplot.colorbar()
pyplot.ylabel('True label')
pyplot.xlabel('Predicted label')
pyplot.show()