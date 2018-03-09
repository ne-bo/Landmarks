import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import script
import os

batch_size = 5
initial_image_size = 586 # for VGG
initial_image_scale_size = 586 # for VGG


def get_filenames_and_labels(data_folder, test_or_train='test'):
    images_paths = []
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []

    keys = list(os.listdir(data_folder + ('%s/' % (test_or_train))))

    print(keys)
    with open('keys_for_%s' % test_or_train, "w") as fout:
        fout.write(" ".join([str(el) for el in keys]))

    i = 0
    for id in keys:
        path = data_folder + ('%s/%s' % (test_or_train, id))
        images_paths.append(path)
        id = (id.replace('.jpg',''))

        if test_or_train == 'train':
            images_labels = train_labels
            train_images.append(path)
            train_labels.append(i)
        else:
            images_labels = test_labels
            test_images.append(path)
            test_labels.append(i)
        i = i + 1

    images_labels = np.array(images_labels)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    print('images_labels.shape ', images_labels.shape)

    train_images = np.array(train_images)
    test_images = np.array(test_images)

    return images_labels, images_paths, train_images, train_labels, test_images, test_labels


class Landmark(Dataset):
    def __init__(self, data_folder, transform=None, test_or_train='test'):
        self.data_folder = data_folder
        self.transform = transform
        if test_or_train == 'train':
            self.train = True
        else:
            self.train = False
        self.images_labels, \
        self.images_paths, \
        self.train_images, \
        self.train_labels, \
        self.test_images, \
        self.test_labels = get_filenames_and_labels(data_folder,
                                                    test_or_train=test_or_train)

        print('self.images_labels ', self.images_labels)

    def __len__(self):
        if self.train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, index):
        #print('index ', index)
        transform_for_correction = transforms.Compose([
            transforms.ToPILImage(),
        ])

        if self.train:
            images_paths = self.train_images
            labels = self.train_labels
        else:
            images_paths = self.test_images
            labels = self.test_labels

        if index > 1089999:
            if os.path.exists(images_paths[index]):
                try:
                    image = self.transform(Image.open(images_paths[index]))
                except:
                    print('images_paths[index] !!!!', images_paths[index])
                label = labels[index]
                #print('label ', label)
            else:
                print('Image %s is not downloaded yet!', images_paths[index])

            if image.shape[0] == 1:
                # print('Grayscale image is found! ', self.images_paths[index])
                image = transform_for_correction(image)
                image = transforms.ImageOps.colorize(image, (0, 0, 0), (255, 255, 255))
                image = self.transform(image)
                # print('new image.shape ', image.shape)

            if image.shape[1] < initial_image_size or image.shape[2] < initial_image_size:
                print('image is too small', image.shape)
        else:
            image, label = torch.from_numpy(np.zeros((1, 1))), labels[index]

        return image, label


def create_transformations_for_test_and_train():
    transform_train = transforms.Compose([
        transforms.Scale(initial_image_scale_size),
        transforms.RandomCrop(initial_image_size, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Scale(initial_image_scale_size),
        transforms.CenterCrop(initial_image_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_test, transform_train


def create_new_train_and_test_datasets(transform_train, transform_test, data_folder):
    # create new dataset for representational learning
    new_train_dataset = Landmark(data_folder=data_folder,
                                 transform=transform_train,
                                 test_or_train='train'
                                 )
    new_test_dataset = Landmark(data_folder=data_folder,
                                transform=transform_test,
                                test_or_train='test'
                                )
    print('len(new_train_dataset.train_images) ', len(new_train_dataset.train_images))
    print('len(new_train_dataset.test_images) ', len(new_train_dataset.test_images))

    print('len(new_test_dataset.train_images) ', len(new_test_dataset.train_images))
    print('len(new_test_dataset.test_images) ', len(new_test_dataset.test_images))

    return new_test_dataset, new_train_dataset


def download_landmark(data_folder):
    transform_train, transform_test = create_transformations_for_test_and_train()
    new_test_dataset, new_train_dataset = create_new_train_and_test_datasets(transform_train, transform_test,
                                                                             data_folder)

    train_loader = data.DataLoader(new_train_dataset,
                                   batch_size=batch_size,
                                   drop_last=False,  
                                   shuffle=False, 
                                   num_workers=8)
    print('train_loader.batch_size = ', train_loader.batch_size,
          ' train_loader.batch_sampler.batch_size =', train_loader.batch_sampler.batch_size,

          ' train_loader.dataset ', train_loader.dataset)
    #print('new_test_dataset.images_paths', new_test_dataset.images_paths)
    print('new_test_dataset.images_labels', new_test_dataset.images_labels)
    print('ful batch size = ', len(new_test_dataset.test_labels))
    test_loader = data.DataLoader(new_test_dataset,
                                  batch_size=batch_size,
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=8)

    print('new_train_dataset ', new_train_dataset.__len__())
    print('new_test_dataset ', new_test_dataset.__len__())
    #print('new_train_dataset.images_paths', new_train_dataset.images_paths)
    print('new_train_dataset.images_labels', new_train_dataset.images_labels)
    print('ful batch size = ', len(new_train_dataset.test_labels))

    return train_loader, test_loader
