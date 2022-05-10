from hmac import trans_36
from socket import TIPC_ADDR_ID
from xml.sax.handler import feature_namespace_prefixes
from matplotlib import pyplot as plt, transforms
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
import pandas as pd
import os
import time
from torchsummary import summary

train_triplets_path = "/home/jonas/code/task3/train_triplets.txt"
test_triplets = np.loadtxt('/home/jonas/code/task3/test_triplets.txt')

food_directory = "/home/jonas/code/task3/food/"
filename_suffix = "jpg"

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

class Triplets(Dataset):
    def __init__(self, triplets_path, img_dir, transform = None):
        self.df = pd.read_csv(triplets_path, sep=' ', header=None, converters={i: str for i in range(0, 3)})
        self.triplets_path = triplets_path
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df.index)

    def ImageFromName(self, image_name):
        # construct path from name 
        image_path = os.path.join(self.img_dir, image_name + str(".") + filename_suffix)
        # load image
        return torchvision.io.read_image(image_path).float()
        
    def DataTensorFromTriple(self, triple):
        triple_tensor = torch.zeros((3,3,224,224),dtype=float) # [3 images, 3 channels, 224x224 px]
        for i in range(0,3):
            image = self.ImageFromName(triple[i])
            if self.transform is not None:
                image_tensor = self.transform(image)
            # put image in corresponding position in tensor
            triple_tensor[i] = image_tensor
        return triple_tensor.int()
        

    def __getitem__(self,index):
        triple = self.df.iloc[index]
        triple_tensor = self.DataTensorFromTriple(triple)
        y = torch.zeros(2)
        # balance dataset
        if bool(random.getrandbits(1)):
            # switch images B and C randomly
            temp = triple_tensor[1,:,:,:].clone()
            triple_tensor[1,:,:,:] = triple_tensor[2,:,:,:].clone()
            triple_tensor[2,:,:,:] = temp.clone()
            y[1] = 1
        else:
            y[0] = 1

        return triple_tensor, y.int()


triplets = Triplets(train_triplets_path, food_directory, train_transform)
data_iter = iter(triplets)

# test dataset
# for j in range(0,5):
#     images, labels = next(data_iter)

#     print(images[0])
#     print(images[0].permute(1,2,0).int())

#     fig = plt.figure(figsize=(8, 8))
#     columns = 3
#     rows = 1
#     for i in range(1, 4):
#         fig.add_subplot(rows, columns, i)
#         plt.imshow(images[i-1].permute(1,2,0).int())
#     plt.show()

# PREPROCESSING

print("Prepocessing..")
batch_size = 64
validation_split = .3


triplets_size = len(triplets)
indices = list(range(triplets_size))
split = int(np.floor(validation_split * triplets_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(triplets, batch_size=64, sampler=train_sampler)
validation_loader = DataLoader(triplets, batch_size=64, sampler=valid_sampler)

#dataloader = DataLoader(triplets, batch_size=64, shuffle= True)

# MODEL

print("Building Model..")

class MORESIMILAR_CNN(torch.nn.Module):
    def __init__(self):
        super(MORESIMILAR_CNN, self).__init__()
        # load pretrained ResNet
        self.res = models.resnet18(pretrained = True)
        # reset final layer, to get image features
        num_ftrs = self.res.fc.in_features
        self.res.fc = nn.Linear(num_ftrs,10)
        # load cosine similarity
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, x1, x2, x3):#, onehot = False): # assume x contains images A, B and C in a tensor
        # compute features for all three images
        print("check")
        feat_A = torch.reshape(self.res(x1.int()), (-1,))#.type(torch.LongTensor))
        print("check")
        feat_B = torch.reshape(self.res(x2), (-1,))
        print("check")
        feat_C = torch.reshape(self.res(x3), (-1,))
        print("check")
        print(feat_A.shape)
        print("check")
        # compute cosine distances of (A,B) and (A,C)
        similarities = torch.zeros(2)
        similarities[0] = self.cos(feat_A, feat_B)
        similarities[1] = self.cos(feat_A, feat_C)
        # if onehot:
        #     similarities = nn.functional.one_hot(similarities)
        return similarities
    
model = MORESIMILAR_CNN()#.cuda()
print(model) # show model

#summary(model,input_size = (3,3,224,224))
# three identical cnns in prallel, sharing weights, resulting in
# equally sized feature vectors. 
# between a,b and a,c, the cosine similariy is computed, be mindful that we
# probably (?) only want positive activations.
# the loss would just be some kind of difference between the preprocessed 
# labels and the cosine similarity.

# TRAIN

loss_fn = torch.nn.CrossEntropyLoss()

def accuracy(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
  # computes the classification accuracy
  correct_label = torch.argmax(logits, axis=-1) == torch.argmax(label, axis=-1)
  assert correct_label.shape == (logits.shape[0],)
  acc = torch.mean(correct_label.float())
  assert 0. <= acc <= 1.
  return acc

def evaluate(model: torch.nn.Module) -> torch.Tensor:
  # goes through the test dataset and computes the test accuracy
  model.eval()  # bring the model into eval mode
  with torch.no_grad():
    acc_cum = 0.0
    num_eval_samples = 0
    for x_batch_test, y_label_test in validation_loader:
      #x_batch_test, y_label_test = x_batch_test.cuda(), y_label_test.cuda()
      batch_size = x_batch_test.shape[0]
      num_eval_samples += batch_size
      acc_cum += accuracy(model(x_batch_test), y_label_test) * batch_size
    avg_acc = acc_cum / num_eval_samples
    assert 0 <= avg_acc <= 1
    return avg_acc

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    # reset statistics trackers
    train_loss_cum = 0.0
    acc_cum = 0.0
    num_samples_epoch = 0
    t = time.time()

    for x_batch, y_batch in train_loader:
        print(x_batch.shape)
        optimizer.zero_grad()
        model.train()

        #x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        print(x_batch[:,0].shape)
        logits = model(x_batch[:,0].int(), x_batch[:,1].int(), x_batch[:,2].int())#, x_batch[:,1])#, x_batch[:,2])
        loss = loss_fn(logits, y_batch)

        loss.backward()
        optimizer.step()

        num_samples_batch = x_batch.shape[0]
        num_samples_epoch += num_samples_batch
        train_loss_cum += loss*num_samples_batch
        acc_cum += accuracy(logits, y_batch) * num_samples_batch

    # average the accumulated statistics
    avg_train_loss = train_loss_cum / num_samples_epoch
    avg_acc = acc_cum / num_samples_epoch
    test_acc = evaluate(model)
    epoch_duration = time.time() - t

    # print some infos
    print(f'Epoch {epoch} | Train loss: {train_loss_cum.item():.4f} | '
    f' Train accuracy: {avg_acc.item():.4f} | Test accuracy: {test_acc.item():.4f} |'
    f' Duration {epoch_duration:.2f} sec')

    # save checkpoint of model
    if epoch % 5 == 0 and epoch > 0:
        save_path = f'model_epoch_{epoch}.pt'
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    save_path)
        print(f'Saved model checkpoint to {save_path}')

# PREDICT

#dataPoint, label = next(data_iter)
#prediction = moresim(dataPoint.type(torch.float))
#print(prediction)

# feed triple images into model, retrieve two cosine similarities
# one hot encode them, or just take the image with a larger similarity
# as image 2 (more similar to A), and the image with the smaller similarity 
# as image 3 (less similar to A)