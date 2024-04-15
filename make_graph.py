import matplotlib.pyplot as plt
import numpy as np

model_name = 'resnet18'
train_loss = np.load(f'result_numpy/{model_name}/train_loss.npy')
train_acc = np.load(f'result_numpy/{model_name}/train_acc.npy')
val_loss = np.load(f'result_numpy/{model_name}/val_loss.npy')
val_acc = np.load(f'result_numpy/{model_name}/val_acc.npy')

# plot Loss Graph
plt.subplot(2, 1, 1)
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(0, 2)
plt.legend(['train', 'validation'], loc='upper left')

# plot Acc Graph
plt.subplot(2, 1, 2)
plt.plot(train_acc * 100)
plt.plot(val_acc * 100)
plt.title('model acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim(0, 100)
plt.legend(['train', 'validation'], loc='upper left')

plt.tight_layout()
plt.savefig(f'graphs/{model_name}.png')
