import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_img_and_intensities(batch, i, inputs):
    """
    Args:
        batch: batch number
        i: integer for slice #
        inputs: shape (batch_size, x,y,z)
    Returns:
        a plot with the image and its intensity histogram
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (12,12))

    img = ax1.imshow(inputs[batch][i].squeeze(), cmap = 'bone')
    plt.colorbar(img, fraction=0.046, pad=0.04, ax= ax1)
    ax1.set_title("Input Image_"+str(i))

    ax2.hist(inputs[batch][i].ravel())
    ax2.set_title("Input Intensities")

def plot_intensity_histograms(batch, i, inputs, labels):
    """
    Args:
        batch: batch number
        i: integer for slice #
        inputs: shape (batch_size, x,y,z)
        labels: same as inputs
    Returns:
        a plot of the input image, label, the corresponding intensity histogram and the positive class intensity histogram
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize= (12,12))

    inputs = inputs[batch][i].squeeze()
    labels = labels[batch][i].squeeze()

    img = ax1.imshow(inputs, cmap = 'bone')
    plt.colorbar(img, fraction=0.046, pad=0.04, ax= ax1)
    ax1.set_title("Input Image_"+str(i))

    ax2.imshow(labels, cmap = 'bone')
    ax2.set_title("Label_"+str(i))

    ax3.hist(inputs.ravel())
    ax3.set_title("Input Intensities")

    ax4.hist(inputs[labels > 0].ravel())
    ax4.set_title("Positive Class Input Intensities")

def show_pred_3d(batch, i, inputs, labels, pred):
    """
    Plots 3D images
    Args:
        batch: batch number
        i: integer for slice #
        inputs: shape (batch_size, x,y,z)
        labels: same as inputs
        pred: outputted segmentation; same shape as inputs
    Returns:
        a plot of the images
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize= (12,12))
    img = ax1.imshow(inputs[batch][i].squeeze(), cmap = 'bone')
    #     # create an axes on the right side of ax. The width of cax will be 5%
    #     # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    #     divider = make_axes_locatable(ax1)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, fraction=0.046, pad=0.04, ax= ax1)
    ax1.set_title("Input Image_"+str(i))

    ax2.imshow(labels[batch][i].squeeze(), cmap = 'bone')
    ax2.set_title("Label_"+str(i))

    ax3.imshow(pred[batch][i].squeeze(), cmap = 'bone')
    ax3.set_title("Predicted Segmentation_"+str(i))

def plot_scan(scan, start_with, show_every, rows=3, cols=3, box = None):
    """
    Plots multiple scans throughout your medical image.
    Args:
        scan: numpy array with shape (x,y,z)
        start_with: slice to start with
        show_every: size of the step between each slice iteration
        rows: rows of plot
        cols: cols of plot
        box:
    Returns:
        a plot of multiple scans from the same image
    """
    fig,ax = plt.subplots(rows, cols, figsize=[3*cols,3*rows])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/cols), int(i%cols)].set_title('slice %d' % ind)
        ax[int(i/cols), int(i%cols)].axis('off')

        #Draw the scan cropping to the provided box
        if box:
            clipScan = scan[ind,box[2]-1:box[5],
                            box[1]-1:box[4]]
            ax[int(i/cols), int(i%cols)].imshow(clipScan,cmap='gray')
        else:
            ax[int(i/cols), int(i%cols)].imshow(scan[ind],cmap='gray')
    plt.show()
