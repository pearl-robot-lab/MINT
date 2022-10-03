import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300

def plot_conditional_entropy(image,conditional_entropy, joint_entropy, entropy):
    # plot the image and the entropy
    # create a subplot
    fig,axes=plt.subplots(3,2)
    # plot the original image
    axes[0,0].imshow(image[0,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    axes[0,1].imshow(image[1,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    # axes[0].set_title('Original image')
    # plot the entropy of the first image
    axes[1,0].imshow(entropy[0].detach().cpu().numpy(),cmap='jet')
    axes[1,0].set_title('Entropy of the first image', fontsize=3, pad=-10)
    # plot the entropy of the second image
    axes[1,1].imshow(entropy[1].detach().cpu().numpy(),cmap='jet')
    axes[1,1].set_title('Entropy of the second image', fontsize=3, pad=-10)
    # plot the joint entropy
    axes[2,0].imshow(joint_entropy[1].detach().cpu().numpy(),cmap='jet')
    axes[2,0].set_title('Joint entropy of the both images', fontsize=3, pad=-10)
    # plot the conditional entropy
    axes[2,1].imshow(conditional_entropy[1].detach().cpu().numpy(),cmap='jet')
    axes[2,1].set_title('Conditional entropy', fontsize=3,pad=-10)
    # remove the axis ticks and the borders
    for ax in axes.flat:
        # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        ax.axis('off')
    plt.tight_layout()
    # save the figure
    # plt.savefig('conditional_entropy.png')
    plt.show()
    
def plot_transport(image,entropy,conditional,source,target,reconstruction,mutual_information):
    # plot the image and the entropy
    # create a subplot
    fig,axes=plt.subplots(3,3,figsize=(15,15))
    # plot the original image
    axes[0,0].imshow(image[0,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    axes[0,1].imshow(image[1,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    axes[0,2].imshow(mutual_information[1].detach().cpu().numpy(),cmap='jet')
    axes[0,2].set_title('mutual information')
    # plot the entropy of the first image
    axes[1,0].imshow(entropy[0].detach().cpu().numpy(),cmap='jet')
    axes[1,0].set_title('Entropy of the first image')
    # plot conditional
    axes[1,1].imshow(conditional[1].detach().cpu().numpy(),cmap='jet')
    axes[1,1].set_title('conditional entropy')
    # plot the entropy of the second image
    axes[1,2].imshow(entropy[1].detach().cpu().numpy(),cmap='jet')
    axes[1,2].set_title('Entropy of the second image')
    # plot the source info
    axes[2,0].imshow(source[1].detach().cpu().numpy(),cmap='jet')
    axes[2,0].set_title('source_info')
    # plot the target info
    axes[2,1].imshow(target[1].detach().cpu().numpy(),cmap='jet')
    axes[2,1].set_title('target_info')
    # plot reconstruction
    axes[2,2].imshow((reconstruction[1]).detach().cpu().numpy(),cmap='jet')
    axes[2,2].set_title('reconstruction')
    # remove the axis ticks and the borders
    for ax in axes.flat:
        ax.axis('off')
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    # save the figure
    # plt.savefig('conditional_entropy.png')
    plt.show()