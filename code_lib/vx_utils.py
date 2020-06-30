import matplotlib.pyplot as plt


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def nop(points):
    return points


def show_loss_acc(history, need_smooth=True):
    smooth_func = smooth_curve if need_smooth else nop

    history_dict = history.history

    loss_values = smooth_func(history_dict['loss'])
    val_loss_values = smooth_func(history_dict['val_loss'])
    acc_values = smooth_func(history_dict['acc'])
    val_acc_values = smooth_func(history_dict['val_acc'])
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    plt.plot(epochs, acc_values, 'y', label='Training acc')
    plt.plot(epochs, val_acc_values, 'g', label='Validation acc')
    plt.title('Training and validation loss/acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Acc')
    plt.legend()
    plt.show()
