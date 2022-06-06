from __future__ import print_function, division

import time
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.hub import load_state_dict_from_url
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import pandas as pd

plt.ion()   # interactive mode
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""## 圖像轉換
### 題目
torchvision.transforms 提供了許多可靠的 API來讓使用者對圖像進行操作，請試著在 data_transforms 當中對訓練集進行轉換(圖像前處理)，當模型訓練到一定程度時，驗證看看使用該方法是否確實對模型準確率造成影響，然後試著解釋使用該轉換方法會對模型訓練產生什麼影響。

* 至少嘗試使用 **五種** 不同的圖像轉換方法，並且找出最佳的方法組合。(使用方法數量為加分bonus的依據)
* 須在報告中註明每一個方法 **在未使用時的準確率**、**使用後的準確率**，並 **說明該方法的目的** 及 **最終最佳組合的準確率**。

### 說明
請在註解區塊中寫入圖像轉換的方法。

"""

data_transforms = {
        'train': transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(0,30)),
            ]),
            transforms.Resize((224,224) ),
            ########在此區塊填入圖像轉換方法########
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.Pad(padding = (40, 40, 40, 40), padding_mode="symmetric"),
            # transforms.RandomRotation((0,30)),
            # transforms.RandomAdjustSharpness(sharpness_factor=2),
            # transforms.ColorJitter(brightness=.2, hue=.1),

            ########################################
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224) ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224,224) ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

"""## 修改模型架構
### 第一題 題目
在本次作業範例中我們使用了CNN來做為整個分類模型的架構。請以第一題中最佳的圖像轉換方法組合，並基於CNN架構增加或減少模型的隱藏層，並觀察修改模型後對原先準確率的影響(即修改模型的意思)

* 至少使用 **三種** 不同隱藏層或不同的修改模型方法(增加或減少模型的隱藏層，並且找出最佳的模型架構。(修改方法多寡為加分bonus的依據)
* 須在報告中註明每一個方法 **在更改前的準確率**、 **更改後的準確率** 及 **最終最佳模型架構的準確率**。

### 說明
* 因為模型有套用預訓練的參數，所以更改模型的方式比較複雜，

* 請勿直接更改現有隱藏層的參數(輸入、輸出大小等等)，請用增加或減少的方式來修改模型架構。

* 請注意並計算各隱藏層可接受的輸出入大小，以免產生資料維度前後層對不上的問題。

"""

class MyCNN(nn.Module):

  def __init__(self, num_classes=1000):
    super(MyCNN, self).__init__()
    self.features = nn.Sequential(
      #============== 在此區塊新增或減少隱藏層 =================
      nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),

      nn.Conv2d(64, 192, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      
      nn.Conv2d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),

      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),

      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      #==========================================================
    )
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 6 * 6, 4096),#全連接層 fully connected
      nn.ReLU(inplace=True),

      nn.Dropout(),#避免過擬和
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      
      # nn.Linear(4096, num_classes), # 原始模型輸出層
    )

    self.myLinear = nn.Linear(4096, num_classes)


  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)

    x = self.myLinear(x)

    return x

class MyCNN2(nn.Module):

  def __init__(self, num_classes=1000):
    super(MyCNN2, self).__init__()
    self.features = nn.Sequential(
      #============== 在此區塊新增或減少隱藏層 =================
      nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      #============================================================
    )
    self.features2 = nn.Sequential(
      #============== 可在此區塊新增隱藏層 =====================
      nn.Conv2d(256,1024,kernel_size = 3, padding = 1), 
      nn.ReLU(inplace=True),

      nn.Conv2d(1024,256,kernel_size = 3, padding = 1), 
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2),
      
      #===========================================================
    )
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.classifier = nn.Sequential(
      #============== 在此區塊新增或減少隱藏層 =================
      nn.Dropout(),
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      # nn.Linear(4096, num_classes), # 原始模型輸出層
      #===========================================================
    )
    self.classifier2 = nn.Sequential(
      #============== 可在此區塊新增隱藏層 =====================
      


      #===========================================================
      nn.Linear(4096, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.features2(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    x = self.classifier(x)
    x = self.classifier2(x)

    return x

class MyCNN3(nn.Module):

  def __init__(self, num_classes=1000):
    super(MyCNN3, self).__init__()
    self.features = nn.Sequential(
      #============== 在此區塊新增或減少隱藏層 =================
      nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      #============================================================
    )
    self.features2 = nn.Sequential(
      #============== 可在此區塊新增隱藏層 =====================
      nn.Conv2d(256,1024,kernel_size = 5, padding = 1), 
      nn.ReLU(inplace=True),

      nn.Conv2d(1024,256,kernel_size = 3, padding = 1), 
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2),
      
      #===========================================================
    )
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.classifier = nn.Sequential(
      #============== 在此區塊新增或減少隱藏層 =================
      nn.BatchNorm1d(256 * 6 * 6,eps=1e-5),
      nn.Linear(256 * 6 * 6, 4096),
      nn.BatchNorm1d(4096,eps=1e-5),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      #===========================================================
    )
    self.classifier2 = nn.Sequential(
      #============== 可在此區塊新增隱藏層 =====================
      nn.BatchNorm1d(4096,eps=1e-5),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.BatchNorm1d(4096,eps=1e-5),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      #===========================================================
      nn.Linear(4096, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.features2(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    x = self.classifier(x)
    x = self.classifier2(x)

    return x

"""## 訓練模型區塊
包含視覺化模型及訓練模型。
"""

def visualize_model(model, device, dataloaders, class_names, num_images=6,what='val'):
  was_training = model.training
  model.eval()
  images_so_far = 0

  plt.figure(figsize=(18,9))

  with torch.no_grad():
    for i, (inputs, labels, path) in enumerate(dataloaders[what]):
      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)

      for j in range(inputs.size()[0]):
        images_so_far += 1

        img_display = np.transpose(inputs.cpu().data[j].numpy(), (1,2,0)) #numpy:CHW, PIL:HWC
        plt.subplot(num_images//2,2,images_so_far),plt.imshow(img_display) #nrow,ncol,image_idx
        plt.title(f'predicted: {class_names[preds[j]]}')
        plt.savefig(os.path.join("./"+ what +".png"))
        if images_so_far == num_images:
            model.train(mode=was_training)
            return
    model.train(mode=was_training)

def imshow(inp, title=None):
  """Imshow for Tensor."""
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  
  #原先Normalize是對每個channel個別做 減去mean, 再除上std
  inp1 = std * inp + mean

  plt.imshow(inp)

  if title is not None:
      plt.title(title)
  plt.pause(0.001)  # pause a bit so that plots are updated
  plt.imshow(inp1)
  if title is not None:
      plt.title(title)


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, criterion, device, dataloaders, dataset_sizes, optimizer, scheduler, num_epochs=25):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  train_loss, valid_loss = [], []
  train_acc, valid_acc = [], []

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels, path in tqdm(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            if phase == 'train':
                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        # print('preds',preds)
        # print('labels',labels.data)
        
      if phase == 'train':
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]
      print('runnint_correct',running_corrects.double())
      print('dataset size',dataset_sizes[phase])

      if phase == 'train':
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
      else:
        valid_loss.append(epoch_loss)
        valid_acc.append(epoch_acc)

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())


  plt.figure(0)
  plt.plot(range(1,num_epochs+1,1), np.array(train_loss), 'r-', label= "train loss") #relative global step
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  # plt.savefig(f"./train_loss.png")

  # plt.figure(1)
  plt.plot(range(1,num_epochs+1,1), np.array(valid_loss), 'b-', label= "eval loss") #--evaluate_during_training True 在啟用eval
  # plt.xlabel('epoch')
  # plt.ylabel('loss')
  plt.legend()
  plt.savefig(f"./loss.png")

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  torch.save(model.state_dict(),"./output/best_model_" + '{:4f}'.format(best_acc) + ".pt")
  
  return model

"""## 訓練參數 (可調整)
* num_epochs: 訓練回合數
* lr: 訓練速度(learning rate)
* batch_size: 批次(batch)大小
"""

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
      # this is what ImageFolder normally returns 
      original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
      # the image file path
      # path = self.imgs[index][0]
      path = os.path.basename(self.imgs[index][0])
      # make a new tuple that includes original and the path
      tuple_with_path = (original_tuple + (path,))
      # print(tuple_with_path)
      return tuple_with_path


num_epochs = 20
lr = 0.001
batch_size = 5

"""## 主函式"""

def main():
  num_workers = 2
  momentum = 0.9

  # 資料集載入 =======================================================================
  data_dir = './training'
#   data_dir = './Cross-validation/Cross-validation-1'
  image_datasets = {
    x: ImageFolderWithPaths(
      os.path.join(data_dir, x),
      data_transforms[x]
    ) 
    for x in ['train', 'val', 'test']
  }
  dataloaders = {
    x: torch.utils.data.DataLoader(
      image_datasets[x], 
      batch_size=batch_size,
      shuffle=False, 
      num_workers=num_workers
    )
    for x in ['train', 'val', 'test']
  }
  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
  class_names = image_datasets['train'].classes
  # print('clsss name',class_names)
  # 資料集載入 =======================================================================

  # 設定 CUDA 環境 =======================================================================
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
  print(f"Using device {device}\n")
  # 設定 CUDA 環境 =======================================================================


  # Get a batch of training data
  inputs, classes, path = next(iter(dataloaders['train']))
  # Make a grid from batch
  out = torchvision.utils.make_grid(inputs)
  torchvision.utils.save_image(out,'./pic.png')

  imshow(out, title=[class_names[x] for x in classes])

  
  # model =======================================================================
  # Origin
  # model_ft = MyCNN(num_classes=219)

  # Try
  # model_ft = MyCNN2(num_classes=219)

   # Try2
  # model_ft = MyCNN3(num_classes=219)

  # Resnet101
  # model_ft = models.resnet101(pretrained=True)
  # num_ftrs = model_ft.fc.in_features
  # model_ft.fc = nn.Linear(num_ftrs,219)

  # wide_resnet101_2
  # model_ft = models.wide_resnet101_2(pretrained=True)
  # num_ftrs = model_ft.fc.in_features
  # model_ft.fc = nn.Linear(num_ftrs,219)

  # ResNeXt-101 32x8
  # model_ft = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
  # num_ftrs = model_ft.fc.in_features
  # model_ft.fc = nn.Linear(num_ftrs,219)

  # vgg_19_bn
  # model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
  # num_ftrs = model_ft.classifier[6].in_features
  # model_ft.classifier[6] = nn.Linear(num_ftrs,219)

  # densenet161
  model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True)
  num_ftrs = model_ft.classifier.in_features
  model_ft.classifier = nn.Linear(num_ftrs,219)


  # TEST
  # model_ft = models.resnet101(pretrained=True)
  # model_ft = models.efficientnet_b0(pretrained=True)
  # model_ft = EfficientNet.from_pretrained('efficientnet-b0')

  #TEST inception v3
  # model_ft.aux_logits = False 
  #TEST squeezenet1_0
  # num_features = model_ft.classifier[1].in_channels
  # features = list(model_ft.classifier.children())[:-3] # Remove last 3 layers
  # features.extend([nn.Conv2d(num_features, 219, kernel_size=1)]) # Add
  # features.extend([nn.ReLU(inplace=True)]) # Add
  # features.extend([nn.AdaptiveAvgPool2d(output_size=(1,1))]) # Add
  # model_ft.classifier = nn.Sequential(*features) # Replace the model classifier

  # print(model_ft) 

  # temp = model_ft.classifier[0:2]
  # model_ft.classifier = temp
  # print(model_ft.classifier)

  # model_ft.classifier[2] = nn.Dropout(p=0.4)
  # model_ft.classifier[5] = nn.Dropout(p=0.4)
  # model_ft.classifier[2] = nn.BatchNorm1d(4096,eps=1e-5)
  # model_ft.classifier[5] = nn.BatchNorm1d(4096,eps=1e-5)

  # print('hihi',model_ft)
  # num_ftrs = model_ft.fc.in_features
  # model_ft.fc = nn.Linear(num_ftrs,219)
  # print(model_ft)

  # pretrained_dict = load_state_dict_from_url(
  #   'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
  #   progress=True
  # )
  # model_dict = model_ft.state_dict()
  # # 1. filter out unnecessary keys
  # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  # # 2. overwrite entries in the existing state dict
  # model_dict.update(pretrained_dict) 
  # # 3. load the new state dict
  # model_ft.load_state_dict(model_dict)

  # for k,v in model_dict.items():
  #   print(k)

  model_ft = model_ft.to(device)
  # model =======================================================================

  parameter_count = count_parameters(model_ft)
  print(f"#parameters:{parameter_count}")
  print(f"batch_size:{batch_size}")


  criterion = nn.CrossEntropyLoss()

  # Observe that all parameters are being optimized
  optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#   model_ft = train_model(
#     model_ft, 
#     criterion, 
#     device, 
#     dataloaders, 
#     dataset_sizes, 
#     optimizer_ft, 
#     exp_lr_scheduler,     
#     num_epochs=num_epochs
#   )
  test_model(model_ft, device, dataloaders, class_names)
  # visualize_model(model_ft, device, dataloaders, class_names)

def test_model(model, device, dataloaders, class_names):

  model.load_state_dict(torch.load('./output/best_model_1.000000.pt'))
  model.eval()

  # was_training = model.training
  images_so_far = 0

  # plt.figure(figsize=(18,9))
  filename = []
  predict_list = []

  with torch.no_grad():
    for i, (inputs, labels, path) in enumerate(dataloaders['test']):
      inputs = inputs.to(device)
      labels = labels.to(device)
      path = path

      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      for j in range(inputs.size()[0]):
        # print(i)
        print(path[j])
        print(f"predicted: {class_names[preds[j]]}")
        filename = np.append(filename, path[j])
        predict_list = np.append(predict_list, class_names[preds[j]])

  answer = {
      "filename": filename,
      "category": predict_list,
  }
  df = pd.DataFrame(answer)
  pd.options.display.float_format = '{:,.0f}'.format
  df.to_csv('./model_1.000000.csv', encoding="utf-8", index = False) 

if __name__ == '__main__':
    main()