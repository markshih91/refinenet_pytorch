import os
import random as rand


def split_train_test(images_folder, train_list_file, test_list_file, train_rate=0.9):
     '''
     split the data into train part and test part
     :param images_folder (str):images folder
     :param train_list_file (str):store the prefix file names of the train part
     :param test_list_file (str):store the prefix file names of the test part
     :param train_rate (double):the Proportion of training data to total data, and the rest is the test data
     :return:None
     '''
     namelist = os.listdir(images_folder)
     namelist = [name[:-4] for name in namelist]
     namelist.sort(key=lambda str:int(str))
     rand.shuffle(namelist)

     train_list_file = open(train_list_file,'w')
     test_list_file = open(test_list_file, 'w')

     for name in namelist[ : int(len(namelist) * train_rate)]:
          train_list_file.write(name + '\n')

     for name in namelist[int(len(namelist) * train_rate) : ]:
          test_list_file.write(name + '\n')


if __name__ == '__main__':

     images_folder = 'nyu_images'
     train_list_file = 'train.txt'
     test_list_file = 'test.txt'

     split_train_test(images_folder, train_list_file, test_list_file)