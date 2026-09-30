# NOTE WORK-IN-PROGRESS
"""
Using https://github.com/Parth47/Contrastive-Learning/blob/main/Contrastive_Learning.py as a starting point, this implements a variant of Chen et al's "A Simple 
# Framework for Contrastive Learning of Visual Representations" (https://arxiv.org/abs/2002.05709), to try to develop a good encoding for the visual features of
# Japanese characters as typically typeset. Will also likely try applying this to visualizations of entire Japanese words/expressions, rather than using
# individual character (kanji, kana) as an intermediary
# For characters, currently we use all chars representable in the NotoSansCJK font as a convenient pool of training examples, & augmenting these with contrastive
# transformations: Gaussian blur, random rotations, random crop, resize. Later, will use JMDict as a source of whole-word/expression training samples.
"""
# NOTE The Parth47 project was a great starting point, but:
#   - its README is incorrect in a few places (e.g., suggests looking for a non-existent Jupyter script)
#   - the custom data-loader class is not present; we've replaced it with our own data-loading approach
#   - since we're not working in Jupyter, we replace tqdm.notebook with normal tqdm
#   - to easily handle images of varying shape (especially important for directly applying the NN to whole words or expressions, which of course vary in length), we
#     implement a fully convolutional neural network (FCN), in particular avoiding the fully-connected linear layer Parth47's repo used. This also creates some
#     choices we're still considering when it comes to output visualizations
#   - although our FCN currently seems to work properly on images of variable size, PyTorch's default batching seems to break, forcing us to a batch size of 1. This
#     makes individual training epochs very noisy; will probably add code to pool parameter updates over multiple epochs, although it would be even better
#     find an alternative batcher that can accept variable-sized images, or write our own
#   - a matter of taste, but have reorganized much of the original script to try to improve modularity & align comments w/ my personal needs
# TODO 9/30/2026 more of a high-level consideration than a concrete to-do, but it would be really nice to try modifying the FCN so that the resulting DNN itself
# directly assigns input images of words/expressions to clusters, to contrast this with our current approaches (which manage the clustering component of our
# workflow to mathematical optimization, k-means, etc). It isn't really obvious where to get very much data on what kinds of words/expressions people find visually
# similar, though; maybe we can freeze the structure of the regular FCN, once trained, and then do something like conditional finetuning / reinforcement-learning to
# determine the parameters of an extra layer for cluster assignment, based on our own personal Anki leech data and clustering preferences?

# Standard libraries
import warnings, os, glob, pathlib
warnings.filterwarnings('ignore')

# PyTorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional

# Non-standard libraries (exlc. PyTorch)
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import umap, fire
import umap.plot
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

def show_images(images, title=''):
    """
        Helper function to display a row of images.
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(9, 3), squeeze=True)
    axes = np.atleast_1d(axes) # When len(images)==1, subscripting below fails o.w.
    for i in range(num_images):
        print(f"Attempting to plot {images[i]} with type: {type(images[i])}")
        img = np.squeeze(images[i]) # Removes 1-d channels, e.g. 1x28x28 -> 28x28
        #img = np.transpose(img, (1, 2, 0))
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    fig.suptitle(title, fontsize=16)
    plt.show()

class FCN(nn.Module):
    """
        Simple fully-convolutional neural network (i.e., which can accept variable-sized input images)
    """
    def __init__(self):
        super(FCN, self).__init__()
        self.conv1 = nn.Sequential(
            # Per https://docs.pytorch.org/docs/2.13/generated/torch.nn.modules.conv.Conv2d.html
            # nn.Conv2d, Output = (In + 2 * Padding - Dilation * (Kernel - 1) - 1)  / Stride + 1
            #   (All variables in relevant dimension, e.g. Height or Width)
            # In our case, defaults Stride = Dilation = 1, Padding = 0, so
            #                   = ((In - 4) - 1) + 1 = In - 4
            # So, with batch size N,
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5), # Output: N x 32 x (H - 4) x (W - 4) 
            nn.BatchNorm2d(32), # Normalize each of the 32 channels (approx. de-mean, normalize by stddev)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: N x 32 x floor([H-4]/2) x floor([W-4]/2),
            nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5), # Output N x 64 x floor([H-4]/2) - 4 x floor([W-4]/2) - 4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x floor[floor([H-4]/2) - 4]/2 x floor[floor([W-4]/2) - 4]/2
            nn.Dropout(0.3)
        )
        #self.final = nn.Sequential(
        #    nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1), # Output N x 1 x floor[floor([H-4]/2) - 4]/2 x floor[floor([W-4]/2) - 4]/2
        #    nn.AdaptiveAvgPool2d(1) # Output N x 1 x 1 x 1
        #)
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# Contrastive Loss: cosine similarity b/w embedding of anchor & contrastive img; use MSE to push this towards a target `distance` value
# positive pair (same character), target=1.0; negative pair (different characters), target=0.0
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        # TODO 9/30/2026 try a margin approach instead (e.g., y d^2 + (1−y) max(0,m−d)^2, rather than trying to force images not in the same
        # class to have completely orthogonal FCN output vectors
        self.similarity = nn.CosineSimilarity(dim=-1, eps=1e-7)

    def forward(self, anchor, contrastive, distance):
        score = self.similarity(anchor, contrastive) # Calculate the cosine similarity score between the two embeddings
        return nn.MSELoss()(score, distance) # Use MSE to compare the score with the target distance (1.0 or 0.0)

def train_model(net, optimizer, trainLoader, scheduler, device, loss_fxn, checkpoint_dir, epoch_count=10):
    """Function to train the model."""
    losses = []
    lrs = []
    net.train() # Set the model to training mode

    for epoch in range(epoch_count):
        print(f"--- Epoch {epoch+1}/{epoch_count} ---")
        epoch_loss = 0
        batches = 0
        
        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        print(f"Learning Rate: {current_lr}")

        # Iterate over the training data
        for anchor, contrastive, distance, _ in tqdm(trainLoader):
            # If we don't rescale contrastive to match anchor, cosine similarity loss will sometimes fail due to shape mismatch
            # Currently we think of this anchor-by-anchor rescaling as just introducing another small contrastive transformation
            # TODO 9/30/2026 alternatively, try using adaptive_avg_pool2d to force FCN output layer to a fixed shape
            contrastive = torch.nn.functional.interpolate(
                contrastive, # This has N, C, H, W like: torch.Size([1, 1, 461, 483]) [# imgs, channels, height, width]
                size=anchor.shape[-2:], # Interpolate expects spatial dimensions only, not # imgs or channels
                mode='bilinear',
                align_corners=False
            )
            batches += 1
             
            anchor, contrastive, distance = anchor.to(device), contrastive.to(device), distance.to(torch.float32).to(device) # Move data to the selected device (..CPU)

            optimizer.zero_grad() # Zero out gradients
            
            anchor_out = net(anchor) # Forward pass: compute embeddings
            contrastive_out = net(contrastive)
            
            loss = loss_fxn(anchor_out, contrastive_out, distance)
            loss.backward() # Backward pass: compute gradients
            optimizer.step() # Update network weights
            epoch_loss += loss.item()

        # Calculate / store average loss for this epoch
        avg_epoch_loss = epoch_loss / batches
        losses.append(avg_epoch_loss)
        print(f"Average Epoch Loss: {avg_epoch_loss:.4f}")
        
        # Step the scheduler
        scheduler.step()

        # Save a checkpoint of the model at the end of each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
        torch.save(net.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    return {
        "net": net,
        "losses": losses
    }

# Optional fxn to load a pre-trained model.
def load_model_from_checkpoint(device, path='checkpoints/model_epoch_99.pt'):
    """Loads a pre-trained model state from a checkpoint file.""" 
    if not os.path.exists(path): # Ensure the checkpoint file exists
        raise FileNotFoundError(f"Checkpoint file not found at {path}. Please train the model first or download the pre-trained weights.")
    checkpoint = torch.load(path, map_location=device)
    model = Network().to(device)
    model.load_state_dict(checkpoint)
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model

# Define a standard transformation pipeline for the images.
# - ToPILImage(): Converts the input data (numpy array) to a PIL Image.
# - ToTensor(): Converts the PIL Image to a PyTorch tensor.
# - Normalize(0.5, 0.5): Normalizes the tensor's pixel values to have a mean of 0.5
#   and a standard deviation of 0.5. This scales the pixel values from [0, 1] to [-1, 1],
#   which helps stabilize training.
default_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

class ContrastiveImageDataset(Dataset):
    def __init__(self, image_dir, transform=default_transform, rng_seed=3141592653, img_suffix="png", num_samples=150000,
                        pos_prob=0.5, contrastive_tuples=None):
        self.rng                = np.random.default_rng(42)
        self.image_dir          = pathlib.Path(image_dir)
        self.transform          = transform
        self.img_suffix         = img_suffix
        self.pos_prob           = pos_prob
        self.num_samples        = num_samples
        self.contrastive_tuples = self.getContrastiveTuples() if contrastive_tuples is None else contrastive_tuples

    def getContrastiveTuples(self):
        contrast_transforms = ["GaussianBlur", "ResizeCrop", "Rotation"]
        anchor_img_paths = glob.glob(str(self.image_dir) + f"/*.{self.img_suffix}")
        anchor_img_chars = [pathlib.Path(p).stem[:pathlib.Path(p).stem.index("_")] for p in anchor_img_paths]
        contrastive_tuples = []
        for index in range(self.num_samples):
            anchor_char = self.rng.choice(anchor_img_chars)
            anchor_img_path = str(self.image_dir) + f"/{anchor_char}_1024.{self.img_suffix}"

            contrast_char = anchor_char if self.rng.random() < self.pos_prob else self.rng.choice(anchor_img_chars)
            contrast_transform = self.rng.choice(contrast_transforms)
            contrast_img_path = str(self.image_dir) + f"/{contrast_transform}/{contrast_char}_1024.{self.img_suffix}"
            
            target_similarity = 1.0 if anchor_char==contrast_char else 0.0 # Aim for orthogonality if characters are distinct (TODO 9/21/2026: try margin-based loss)
            label = "Positive" if anchor_char==contrast_char else "Negative"
            
            contrastive_tuples.append( (anchor_img_path, contrast_img_path, target_similarity, label) )
        return contrastive_tuples

    def __len__(self):
        return len(self.contrastive_tuples)

    def __getitem__(self, index):
        anchor_path, contrastive_path, target_similarity, label = self.contrastive_tuples[index]

        anchor_image = Image.open(anchor_path).convert("RGB")
        contrastive_image = Image.open(contrastive_path).convert("RGB")
        if self.transform:
            anchor_image = self.transform(anchor_image)
            contrastive_image = self.transform(contrastive_image)

        return anchor_image, contrastive_image, target_similarity, label

def main(
            train=True, total_num_samples=150000, validation_prob=0.1, training_epochs=10, num_workers=2, prefetch_factor=100, 
            training_batch_size=1, validation_batch_size=1, show_input_visualizations=False, show_output_visualizations=False,
            checkpoint_dir = 'checkpoints/'
        ):
    # TODO NOTE 9/30/2026 an unexpected hiccup with building an FCN: PyTorch's batcher itself does not behave well with images of different sizes, which has forced
    # us to use training_batch_size = validation_batch_size = 1. Currently this means SGD updates are highly unstable; will add manual pooling of SGD parameter
    # updates, and maybe later explore whether we can extend the PyTorch batcher's behavior, or look for an alternative batcher that is more FCN-friendly
    validation_start_ind    = int(total_num_samples*(1-validation_prob))
    base_dataset            = ContrastiveImageDataset("notosans_cjk_chars_as_imgs/", num_samples=total_num_samples)
    training_dataset        = ContrastiveImageDataset("notosans_cjk_chars_as_imgs/", contrastive_tuples=base_dataset.contrastive_tuples[:validation_start_ind])
    validation_dataset      = ContrastiveImageDataset("notosans_cjk_chars_as_imgs/", contrastive_tuples=base_dataset.contrastive_tuples[validation_start_ind:])
    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    trainLoader = DataLoader(
        training_dataset,
        batch_size=training_batch_size, # # samples in each batch
        shuffle=True,                   # Randomly shuffle data at start of every epoch
        pin_memory=False,               # NOTE we're training w/ CPU-only locally on an old laptop (or on free-tier, GPU-less cloud resources)
        num_workers=num_workers,        # # of subprocesses used when data loading (NOTE: free-tier cloud VMs only have 2 threads available)
        prefetch_factor=prefetch_factor # # pre-loaded batches
    )

    valLoader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,   # Don't need e.g. gradients in RAM for validation, so batch can be bigger, but uneven img sizes break DataLoader batching
        shuffle=False,                      # No benefit from randomizing order of validation data, since not used for SGD training
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )

    if show_input_visualizations:
        # Quick visualization of 1st training batch
        anchor_images, contrastive_images, target_similarities, labels = next(iter(trainLoader))
        print("Visualizing first batch of data:")
        print(f"\tAnchor labels: {labels[:4]}")
        print(f"\tDistance labels (1=Positive, 0=Negative): {target_similarities[:4].numpy()}")
        # Display the first 4 samples from the batch
        show_images(anchor_images[:4].numpy(), title='Anchor Images')
        show_images(contrastive_images[:4].numpy(), title='Contrastive Samples (+/- Examples)')

    import os
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    assert device == 'cpu', f"torch unexpectedly thinks a CUDA-compatible GPU is available. (worth investigating?)"

    net = FCN().to(device) # NN / loss fxn / optimizer instantiation
    loss_fxn = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)
 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3) # Decreases learning rate over time
 
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if train:
        training_result = train_model(net, optimizer, trainLoader, scheduler, device, loss_fxn, checkpoint_dir, epoch_count=training_epochs)
        model = training_result["net"]
    else:
        model = load_model_from_checkpoint(device) # Will fail if 'model_epoch_99.pt' file not in checkpoints/

    from IPython.display import Image, display # Graph loss curve
    if train:
        print("Displaying training-session loss curve:")
        plt.figure(figsize=(10, 5))
        plt.plot(training_result["losses"])
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.grid(True)
        plt.show()
    else:
        if os.path.exists("images/loss-curve.png"):
            print("Displaying pre-saved loss curve from checkpoint-loaded model's training:")
            display(Image(filename="images/loss-curve.png", height=400))
        else:
            print("Pre-saved loss curve image not found (images/loss-curve.png is likely missing?).")

    if show_output_visualizations: # TODO 9/30/2026 wrap this in a fxn
        # Embedding visualization: use PCA and UMAP to project NN embeddings to 2D and 3D, to see if these appear meaningful
        print("Generating embeddings for the training dataset...") # Pass training dataset through trained model to get embeddings for each img
        encoded_data, labels = [], []
        model.eval() # evaluation only, no longer training

        with torch.no_grad(): # Disable gradient calculations for inference
            for anchor, _, _, label in tqdm(trainLoader):
                output = model(anchor.to(device))
                encoded_data.extend(output.cpu().numpy())
                labels.extend(label)

        # NOTE TODO 9/30/2026 this currently fails b/c the FCN outputs are naturally of varying shapes, so np.array is not certain how to construct
        # an array from encoded_data. The simplest fix for this would be using adaptive_avg_pool2d to force FCN output layer to a fixed shape (or
        # using the torch.nn.functional equivalent before running PCA, without altering the catual network), but actively considering options here
        # Our clustering exercise is itself similar to the original goal of this use of PCA/UMAP, so we may just write a secondary version of the FCN
        # that can be run w/ fixed-shape outputs, and just disable these visualizations when the current non-fixed-output-shape variant of the FCN is used
        encoded_data = np.array(encoded_data)
        labels = np.array(labels)
        print(f"Generated {encoded_data.shape[0]} embeddings of dimension {encoded_data.shape[1]}.")

        # Dimensionality Reduction: apply PCA to reduce the embeddings from variable-sized FCN outputs to 3D
        pca = PCA(n_components=3)
        encoded_data_3d = pca.fit_transform(encoded_data)

        # Interactive 3D Scatter Plot (w/ PCA): visualize 3D embeddings. If training succeeded, we expect distinct color clusters (e.g., 1 per character)
        print("Generating interactive 3D scatter plot...")
        scatter = go.Scatter3d(
            x=encoded_data_3d[:, 0],
            y=encoded_data_3d[:, 1],
            z=encoded_data_3d[:, 2],
            mode='markers',
            marker=dict(size=3, color=labels, colorscale='Viridis', opacity=0.8),
            text=labels,
            hoverinfo='text',
        )

        layout = go.Layout(
            title="3D Scatter Plot of Embeddings (PCA-reduced)",
            scene=dict(xaxis=dict(title="PC1"), yaxis=dict(title="PC2"), zaxis=dict(title="PC3")),
            width=900, height=700
        )

        fig = go.Figure(data=[scatter], layout=layout)
        fig.show()

        # 2D Scatter Plot (w/ UMAP): UMAP is often better at preserving local structure than PCA (although has no global-structure guarantee). 
        print("Applying UMAP (cosine metric) to reduce dimensionality to 2D...")
        mapper_cosine = umap.UMAP(random_state=42, metric='cosine').fit(encoded_data)
        umap.plot.points(mapper_cosine, labels=labels, theme='fire')
        plt.title("2D UMAP Visualization (Cosine Metric)")
        plt.show()

        print("Applying UMAP (euclidean metric) to reduce dimensionality to 2D...")
        mapper_euclidean = umap.UMAP(random_state=42, metric='euclidean').fit(encoded_data)
        umap.plot.points(mapper_euclidean, labels=labels, theme='fire')
        plt.title("2D UMAP Visualization (Euclidean Metric)")
        plt.show()

if __name__ == "__main__":
    """
    Example run cmds:

        python contrastive_learner.py main --train=True --total_num_samples=150000 --validation_prob=0.1 --training_epochs=10
    """
    fire.Fire()
