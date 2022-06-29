import matplotlib.pyplot as plt
import numpy as np


def visualize_cost(loss_train, acc_train, loss_val, acc_val):
    num_epochs = len(acc_val)
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.subplot(1, 2, 1)

    plt.plot(range(1, num_epochs + 1), np.array(loss_train), '-o', label='train', linewidth=2)
    plt.plot(range(1, num_epochs + 1), np.array(loss_val), '-o', label='val', linewidth=2)
    plt.xlabel('$Epochs$', size=20)
    plt.ylabel('$Loss$', size=20)
    plt.legend(loc='best', fontsize=20)

    plot2 = plt.subplot(1, 2, 2)
    plot2.plot(range(1, num_epochs + 1), np.array(acc_train), '-o', label='train', linewidth=2)
    plot2.plot(range(1, num_epochs + 1), np.array(acc_val), '-o', label='val', linewidth=2)
    plot2.set_xlabel('$Epochs$', size=20)
    plot2.set_ylabel('$Acc$', size=20)
    plot2.legend(loc='best', fontsize=20)
    plot2.grid(True)

    plt.show()



#call it like that     Visualize.visualize_cost(model.train_loss, model.train_acc, model.val_loss, model.val_acc)
