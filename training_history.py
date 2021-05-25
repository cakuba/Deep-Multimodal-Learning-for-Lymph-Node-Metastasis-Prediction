import pickle
import matplotlib.pyplot as plt

def plot_training_history(history, is_show=True, filename="training_history.png"):
    """
        To visualize the training history of a DL model, including

        acc / val_acc
        loss / val_loss
    """

    if "acc" not in history:
        print("WARNING: cannot find training history information!")
        return None
    
    acc = history['acc']
    val_acc = history['val_acc']

    loss = history['loss']
    val_loss = history['val_loss']

    x = range(len(acc))
    plt.figure(figsize=(8,8))
    plt.subplot(211)
    plt.plot(x,acc,'r')
    plt.plot(x,val_acc,'b')
    plt.grid(True)
    plt.xlim([0, len(acc)])
    plt.title("Accuracy for training (RED) and validation (BLUE) set")

    plt.subplot(212)
    plt.plot(x,loss,'r')
    plt.plot(x,val_loss,'b')
    plt.grid(True)
    plt.xlim([0, len(acc)])
    plt.title("Loss for training (RED) and validation (BLUE) set")

    if is_show:
        plt.show()
    else:
        plt.savefig(filename) 

def vis_history(history_path, is_show=True, filename="training_history.png"):
    """
    to visualize the training history
    """

    is_show = is_show
    filename = filename

    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    plot_training_history(history, is_show, filename)
    