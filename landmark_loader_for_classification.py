import datetime
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import utils
from sampling import UniformSampler
from torch.utils.data.sampler import BatchSampler

batch_size = 46
initial_image_size = 224  # for resnet
initial_image_scale_size = 224  # for resnet


def get_filenames_and_labels(data_folder, test_or_train='test'):
    images_paths = []
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []

    keys = list(os.listdir(data_folder + ('%s/' % (test_or_train))))
    for i, id in enumerate(keys):
        keys[i] = id.replace('.jpg', '')
    keys = np.array(keys)
    print('keys ', keys)

    print('data_folder + (s (test_or_train)', data_folder + ('%s/' % (test_or_train)))
    print(keys[:10])
    with open('classification_keys_for_%s' % test_or_train, "w") as fout:
        fout.write(" ".join([str(el) for el in keys]))

    all_keys, all_labels = utils.get_all_keys_and_labels('/media/natasha/Data/Landmark Classification/train.csv')
    all_keys = np.array(all_keys)
    all_labels = np.array(all_labels)
    print('all_keys ', all_keys[:10])
    print('all_labels ', all_labels[:10])
    print('np.unique(all_labels)', np.unique(all_labels).shape)

    indices = np.where(np.in1d(all_keys, keys))[0]
    print('indices ', indices)
    print('indices[41]', indices[41], 'all_keys[indices[41]] ', all_keys[indices[41]])
    print('indices.shape ', indices.shape, 'all_keys.shape ', all_keys.shape, 'keys.shape', keys.shape)
    print('np.where(keys == 6e815d2054869066)', np.where(keys == '6e815d2054869066'))
    i = 0
    for index in indices:
        id = all_keys[index]
        path = data_folder + ('%s/%s.jpg' % (test_or_train, id))
        images_paths.append(path)
        label = all_labels[index]

        if test_or_train == 'train':
            images_labels = train_labels
            train_images.append(path)
            train_labels.append(label)
        else:
            images_labels = test_labels
            test_images.append(path)
            test_labels.append(0)

    images_labels = np.array(images_labels)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    print('images_labels.shape ', images_labels.shape)

    train_images = np.array(train_images)
    test_images = np.array(test_images)

    # np.save('images_labels', images_labels)
    # np.save('images_paths', images_paths)
    # np.save('train_images', train_images)
    # np.save('train_labels', train_labels)
    # np.save('test_images', test_images)
    # np.save('test_labels', test_labels)
    return images_labels, images_paths, train_images, train_labels, test_images, test_labels


class LandmarkClassification(Dataset):
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
        # print('index ', index)
        transform_for_correction = transforms.Compose([
            transforms.ToPILImage(),
        ])

        if self.train:
            images_paths = self.train_images
            labels = self.train_labels
        else:
            images_paths = self.test_images
            labels = self.test_labels

        if index > -1:
            if os.path.exists(images_paths[index]):
                try:
                    image = self.transform(Image.open(images_paths[index]))
                except:
                    print('images_paths[index] !!!!', images_paths[index])
                label = labels[index]
                # print('label ', label)
            else:
                print('Image %s is not downloaded yet!' % images_paths[index])
                print('index ', index, 'label ', labels[index])

            if image.shape[0] == 1:
                print('Grayscale image is found! ', self.images_paths[index])
                image = transform_for_correction(image)
                image = transforms.ImageOps.colorize(image, (0, 0, 0), (255, 255, 255))
                image = self.transform(image)
                print('new image.shape ', image.shape)

            if image.shape[1] < initial_image_size or image.shape[2] < initial_image_size:
                print('image is too small', image.shape)
        else:
            image, label = torch.from_numpy(np.zeros((1, 1))), labels[index]

        return image, label


#################################################
#
# For resized dataset we don't need scaling transformation!!!
#
#################################################
def create_transformations_for_test_and_train():
    transform_train = transforms.Compose([
        transforms.RandomCrop(224, padding=0),  # for ResNet-50
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(224),  # for ResNet-50
        transforms.ToTensor(),
    ])
    return transform_test, transform_train


def create_new_train_and_test_datasets(transform_train, transform_test, data_folder):
    # create new dataset for representational learning
    new_train_dataset = LandmarkClassification(data_folder=data_folder,
                                               transform=transform_train,
                                               test_or_train='train'
                                               )
    # new_test_dataset = LandmarkClassification(data_folder=data_folder,
    #                                          transform=transform_test,
    #                                          test_or_train='test'
    #                                          )
    print('len(new_train_dataset.train_images) ', len(new_train_dataset.train_images))
    print('len(new_train_dataset.test_images) ', len(new_train_dataset.test_images))

    # print('len(new_test_dataset.train_images) ', len(new_test_dataset.train_images))
    # print('len(new_test_dataset.test_images) ', len(new_test_dataset.test_images))
    new_test_dataset = None
    return new_test_dataset, new_train_dataset


def download_landmark_for_classification(data_folder, uniform_sampling=False):
    transform_train, transform_test = create_transformations_for_test_and_train()
    new_test_dataset, new_train_dataset = create_new_train_and_test_datasets(
        transform_train,
        transform_test,
        data_folder)

    if uniform_sampling:
        number_of_samples_with_the_same_label_in_the_batch = (batch_size + 1) / 2
        train_loader = data.DataLoader(
            new_train_dataset,
            batch_sampler=BatchSampler(
                sampler=UniformSampler(
                    new_train_dataset,
                    batch_size=batch_size,
                    number_of_samples_with_the_same_label_in_the_batch=number_of_samples_with_the_same_label_in_the_batch
                ),
                batch_size=batch_size,
                drop_last=False
            ),
            num_workers=8
        )
    else:
        train_loader = data.DataLoader(new_train_dataset,
                                       batch_size=batch_size,
                                       drop_last=False,
                                       shuffle=False,
                                       num_workers=8)

    print('train_loader.batch_size = ', train_loader.batch_size,
          ' train_loader.batch_sampler.batch_size =', train_loader.batch_sampler.batch_size,

          ' train_loader.dataset ', train_loader.dataset)
    # print('new_test_dataset.images_paths', new_test_dataset.images_paths)
    # print('new_test_dataset.images_labels', new_test_dataset.images_labels)
    # print('ful batch size = ', len(new_test_dataset.test_labels))
    test_loader = None
    # test_loader = data.DataLoader(new_test_dataset,
    #                              batch_size=batch_size,
    #                              drop_last=False,
    #                              shuffle=False,
    #                              num_workers=8)

    # print('new_train_dataset ', new_train_dataset.__len__())
    # print('new_test_dataset ', new_test_dataset.__len__())
    # print('new_train_dataset.images_paths', new_train_dataset.images_paths)
    # print('new_train_dataset.images_labels', new_train_dataset.images_labels)
    # print('ful batch size = ', len(new_train_dataset.test_labels))

    return train_loader, test_loader
