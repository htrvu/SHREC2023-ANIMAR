import matplotlib.pyplot as plt

def plot_logs(training_losses, val_losses, NNs, P10s, NDCGs, mAPs, output_path):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title('Nearest Neighbor')
    plt.plot(NNs)
    plt.xlabel('Epoch')
    
    plt.subplot(2, 3, 2)
    plt.title('Precision at 10')
    plt.plot(P10s)
    plt.xlabel('Epoch')

    plt.subplot(2, 3, 3)
    plt.title('Training loss')
    plt.plot(training_losses)
    plt.xlabel('Epoch')

    plt.subplot(2, 3, 4)
    plt.title('NDCGs')
    plt.plot(NDCGs)
    plt.xlabel('Epoch')

    plt.subplot(2, 3, 5)
    plt.title('mAP')
    plt.plot(mAPs)
    plt.xlabel('Epoch')

    plt.subplot(2, 3, 6)
    plt.title('Val loss')
    plt.plot(val_losses)
    plt.xlabel('Epoch')

    plt.tight_layout()

    plt.savefig(output_path)