from PIL import Image, UnidentifiedImageError
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
import torch
import tarfile
import io
import pandas as pd

from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class T4SADataset(Dataset):
    def __init__(
        self,
        text_dir="T4SA/raw_tweets_text.csv",
        img_dir="T4SA/b-t4sa_imgs.tar",
        labels_dir="T4SA/t4sa_text_sentiment.tsv",
        transform=None,
    ):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

        # Load texts from CSV file
        df = pd.read_csv(text_dir, sep=",", index_col="id")
        df.index = df.index.astype(str)
        self.texts = df["text"].to_dict()
        
        # Open tar file containing images
        self.img_dir = img_dir
        tf = tarfile.open(img_dir)
        self.img_name_map = {}
        available_img_ids = set()
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith('.jpg'):
                base = member.name.split('/')[-1]
                tweet_id = base.split('-')[0]
                self.img_name_map.setdefault(tweet_id, []).append(member)
                available_img_ids.add(tweet_id)
        tf.close()
        self.tf = None


        # Load labels from TSV file
        labels_df = pd.read_csv(labels_dir, sep="\t", index_col="TWID")
        labels_df.index = labels_df.index.astype(str)
        
        # Find intersection of all IDs
        text_ids = set(df.index)
        label_ids = set(labels_df.index)
        
        # Filter to keep only IDs present in all three sources (Text, Image, Label)
        self.valid_ids = list(text_ids & label_ids & available_img_ids)
        
        # Align labels with valid_ids
        self.labels_df = labels_df.loc[self.valid_ids]
        self.labels = self.labels_df[["NEG", "NEU", "POS"]].values

    def get_image_from_tar(self, id):
        """
        Gets a image by a name gathered from file list csv file

        :param name: name of targeted image
        :return: a PIL image
        """
        if self.tf is None:
            self.tf = tarfile.open(self.img_dir)

        paths = self.img_name_map.get(str(id))
        if not paths:
            raise FileNotFoundError(f"Image for tweet ID {id} not found in tar archive {self.img_dir}")
        member = paths[0]  # Use the first image for this tweet ID
        image = self.tf.extractfile(member)
        if image is None:
            raise FileNotFoundError(f"Image {member.name} not found in tar archive {self.img_dir}")

        image = image.read()
        image = Image.open(io.BytesIO(image)).convert('RGB')
        return image

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.valid_ids)

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """
        while True:
            try:
                # if index == (self.__len__() - 1):  # close tarfile opened in __init__
                #    self.tf.close()
                
                # Get the ID for this index
                id = self.valid_ids[index]
                
                # get image
                image = self.get_image_from_tar(id)
                
                if self.transform:
                    image = self.transform(image)

                # get text
                text = self.texts[id]
                soft_label = self.labels[index]
                class_idx = int(soft_label.argmax())  # 0=NEG, 1=NEU, 2=POS
                
                sample = {
                    "image": image,
                    "text": text,
                    "label": class_idx,
                    "soft_label": soft_label,
                }

                return sample
            except (UnidentifiedImageError, OSError, IndexError, Image.DecompressionBombError):
                # print(f"Warning: Skipping corrupted image at index {index}: {e}")
                index = (index + 1) % len(self.valid_ids)


if __name__ == "__main__":
    dataset = T4SADataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    print(f"Dataset size: {len(dataset)}")

    # Test loading a batch
    batch = next(iter(loader))
    images = batch["image"]
    print(f"Batch image tensor shape: {images.shape}")
    print("✓ T4SA Dataset loaded successfully!")

    strict_classes = dataset.labels_df[["NEG", "NEU", "POS"]].values.argmax(axis=1)
    class_names = ["Negative", "Neutral", "Positive"]
    counts = pd.Series(strict_classes).value_counts().sort_index()
    total_samples = sum(counts)
    percentages = [c / total_samples * 100 for c in counts]
    for cls, count, pct in zip(class_names, counts, percentages):
        print(f"{cls}: {count} ({pct:.2f}%)")

    # Analyze distribution of sentiment labels
    plt.figure(figsize=(8, 6))
    bars = plt.bar(class_names, counts, color=["red", "gray", "green"])
    plt.title("Sentiment Class Distribution (T4SA Dataset)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\\n({height/total_samples*100:.1f}%)',
                ha='center', va='bottom')
                
    plt.savefig('t4sa_distribution.png')
    print("Saved t4sa_distribution.png")

        # 2. Visualize Random Samples
    print("\nVisualizing random samples...")
    # Use a small batch size to get a few samples
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    
    images = batch['image']
    texts = batch['text']
    labels = batch['label']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        # Image is [C, H, W], need [H, W, C]
        img = images[i].permute(1, 2, 0).numpy()
        
        # Label info
        label = labels[i]
        label_str = f"Neg: {label[0]:.2f}, Neu: {label[1]:.2f}, Pos: {label[2]:.2f}"
        dominant_idx = torch.argmax(label).item()
        dominant = class_names[dominant_idx] # type: ignore
        
        ax.imshow(img)
        ax.set_title(f"Label: {dominant}\\n{label_str}", fontsize=10)
        ax.axis('off')
        
        # Wrap text for display
        text_content = texts[i]
        # Simple wrapping
        wrapped_text = ""
        words = text_content.split()
        line = ""
        for word in words:
            if len(line) + len(word) > 40:
                wrapped_text += line + "\\n"
                line = word + " "
            else:
                line += word + " "
        wrapped_text += line
        
        ax.text(0.5, -0.05, wrapped_text, ha='center', va='top', transform=ax.transAxes, fontsize=9, wrap=True)

    plt.tight_layout()
    plt.savefig('t4sa_visualization.png')
    print("Saved t4sa_visualization.png")


