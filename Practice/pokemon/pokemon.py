import torch
import os, glob
import random, csv

from torch.utils.data import Dataset



class Pokemon(Dataset):

    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {} # "squirtle": 0
        for name in os.listdir(os.path.join(root)):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        print(self.name2label)

        # image, label
        self.load_csv('images.csv')

    def load_csv(self, filename):

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


    def __len__(self):
        pass


    def __getitem__(self, idx):
        pass


def main():

    db = Pokemon('D:\Projects\pokeman', 224, 'train')



if __name__ == '__main__':
    main()