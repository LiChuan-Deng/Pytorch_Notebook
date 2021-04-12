import torch
import os, glob
import random, csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



class Pokemon(Dataset):

    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {} # "squirtle": 0
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)

        # image, label
        self.images, self.labels = self.load_csv('images.csv')

        if mode == 'train': # 60%
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode == 'val': # 20%
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):

            images = []
            for name in self.name2label.keys():
                # 'pokemon\\bulbasaur\\00000000.png'
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            print(len(images), images)

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        images, labels = [],[]
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)

        return images, labels


    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std + mean
        return x

    def __getitem__(self, idx):
        # idx [0, len(images)]
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'), # string path => image data
            transforms.Resize((self.resize, self.resize)),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label





def main():


    db = Pokemon('D:\Projects\pokeman', 224, 'train')
    x,y = next(iter(db))
    print('sample:', x.shape, y.shape, y)
    loader = DataLoader(db, batch_size=32, shuffle=True)

if __name__ == '__main__':
    main()