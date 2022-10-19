
from myDatasets import  cifar10_dataset, MyDataset
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import DatasetFolder

import numpy as np 
import time
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

from cfg import EfficientNet_cfg as cfg

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)
    print("End of saving !!!")


def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")



## TO DO ##
def plot_learning_curve(x, y):
    """_summary_
    The function is mainly to show and save the learning curves. 
    input: 
        x: data of x axis 
        y: data of y axis 
    output: None 
    """
    #############
    ### TO DO ### 
    # You can consider the package "matplotlib.pyplot" in this part.
    print('plotting the figure!!')
    plt.figure(0)
    plt.plot(x, y[0])
    plt.title('training acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.figure(1)
    plt.plot(x, y[1])
    plt.title('training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.figure(2)
    plt.plot(x, y[2])
    plt.title('validation acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.figure(3)
    plt.plot(x, y[3])
    plt.title('validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.show()
    

def train(model, train_loader, val_loader, num_epoch, log_path, save_path, device, 
          criterion, scheduler, optimizer, train_set):
    start_train = time.time()

    overall_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_acc = np.zeros(num_epoch ,dtype = np.float32)
    overall_val_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_val_acc = np.zeros(num_epoch ,dtype = np.float32)

    best_acc = 0

    for i in range(num_epoch):
        print(f'epoch = {i}')
        # epcoch setting
        start_time = time.time()
        train_loss = 0.0 
        corr_num = 0

        # training part
        # start training

        if i in [17, 20, 23]:
            if not os.path.exists(cfg['unlabeled_data_root'] + '/00'):
                os.chdir('./p2_data/unlabeled')
                os.mkdir('00')
                os.system('cp *.jpg ./00')

            means = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
            unlabeled_set = DatasetFolder("./p2_data/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_transform)
            # Obtain pseudo-labels for unlabeled data using trained model.
            pseudo_set = get_pseudo_labels(unlabeled_set, model)
            print('# of unlabeled dataset:', len(pseudo_set))

            # Construct a new dataset and a data loader for training.
            # This is used in semi-supervised learning only.
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=8, pin_memory=True)


        model.train()
        for batch_idx, ( data, label,) in enumerate(tqdm(train_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            label = label.to(device)

            # pass forward function define in the model and get output 
            output = model(data) 

            # calculate the loss between output and ground truth
            loss = criterion(output, label)
            
            # discard the gradient left from former iteration 
            optimizer.zero_grad()

            # calcualte the gradient from the loss function 
            loss.backward()
            
            # if the gradient is too large, we dont adopt it
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            
            # Update the parameters according to the gradient we calculated
            optimizer.step()

            train_loss += loss.item()

            # predict the label from the last layers' output. Choose index with the biggest probability 
            pred = output.argmax(dim=1)
            
            # correct if label == predict_label
            corr_num += (pred.eq(label.view_as(pred)).sum().item())

        # scheduler += 1 for adjusting learning rate later
        scheduler.step()
        
        # averaging training_loss and calculate accuracy
        train_loss = train_loss / len(train_loader.dataset) 
        train_acc = corr_num / len(train_loader.dataset)
                
        # record the training loss/acc
        overall_loss[i], overall_acc[i] = train_loss, train_acc
        
        ## TO DO ##
        # validation part 
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            corr_num = 0
            val_acc = 0.0
            
            ## TO DO ## 
            # Finish forward part in validation. You can refer to the training part 
            # Note : You don't have to update parameters this part. Just Calculate/record the accuracy and loss. 
            for batch_idx, (data, label,) in enumerate(tqdm(val_loader)):
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                pred = output.argmax(dim=1)

                loss = criterion(output, label)
                val_loss += loss.item()                
                corr_num += (pred.eq(label.view_as(pred)).sum().item())
            
            val_loss = val_loss / len(val_loader.dataset) 
            val_acc = corr_num / len(val_loader.dataset)

            overall_val_loss[i], overall_val_acc[i] = val_loss, val_acc
        
        # Display the results
        end_time = time.time()
        elp_time = end_time - start_time
        min = elp_time // 60 
        sec = elp_time % 60
        print('*'*10)
        #print(f'epoch = {i}')
        print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        print(f'training loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}' )
        print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' )
        print('========================\n')

        with open(log_path, 'a') as f :
            f.write(f'epoch = {i}\n', )
            f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val loss : {val_loss}  val acc = {val_acc}\n' )
            f.write('============================\n')

        # save model for every epoch 
        #torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{i}.pt'))
        
        # save the best model if it gain performance on validation set
        if  val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))

    x = range(0, num_epoch)
    overall_acc = overall_acc.tolist()
    overall_loss = overall_loss.tolist()
    overall_val_acc = overall_val_acc.tolist()
    overall_val_loss = overall_val_loss.tolist()
    # Plot Learning Curve
    ## TO DO ##
    # Consider the function plot_learning_curve(x, y) above
    plot_learning_curve(x, (overall_acc, overall_loss, overall_val_acc, overall_val_loss))

def get_pseudo_labels(dataset, model, threshold=0.999):
    print('Start semi-supervised learning!!')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct a data loader.
    data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    _imgs = []
    _labels = []

    # Iterate over the dataset by batches.
    for batch in tqdm(data_loader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        for _i ,  _p in zip(img, probs):
            if torch.max(_p).item() >= threshold:
                _imgs.append(_i.numpy())
                _labels.append(torch.argmax(_p).item())

    _imgs = np.asarray(_imgs)
    _labels = np.asarray(_labels)
   
    dataset = MyDataset(_imgs, _labels)
    print(len(dataset))
   
    # Turn off the eval mode.
    model.train()
    return dataset
