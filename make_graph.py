import matplotlib.pyplot as plt
import numpy as np

graph_name = 'ResNet vs Wide ResNet'

# model_name = 'resnet18'
# train_acc = np.load(f'result_numpy/{model_name}/train_acc.npy')
#
# model2_name = 'resnet34'
# train2_acc = np.load(f'result_numpy/{model2_name}/train_acc.npy')

models_name = ['resnet34', 'wide_resnet34']

# plot Loss Graph
# plt.subplot(2, 1, 1)
# plt.plot(train_loss)
# plt.plot(val_loss)
# plt.title('model loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.ylim(0, 2)
# plt.legend(['train', 'validation'], loc='upper left')

# plot Acc Graph
# plt.subplot(2, 1, 2)
for model_name in models_name:
    train_acc = np.load(f'result_numpy/{model_name}/train_acc.npy')
    plt.plot(1-train_acc)

# plt.plot(1-train_acc)
# plt.plot(1-train2_acc)
plt.yscale('log', base=10)
plt.ylim(1e-2, 1)
plt.title(graph_name)
plt.xlabel('epoch')
plt.ylabel('error rate')
plt.legend(models_name, loc='upper right')

plt.tight_layout()
plt.savefig(f'graphs/{graph_name}.png')
