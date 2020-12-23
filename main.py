# coding = <utf-8>

import sys
import os 
import os.path as osp 
sys.path.insert(0, osp.abspath(__file__))
import logging 
from math import sqrt, log
import random 
import argparse
from pathlib import Path
import time 


import coloredlogs
from tqdm import tqdm 
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt
import torch 

from utils import model

coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")

savePath = osp.join(os.getcwd(), "outputs")
SAVE_DIR = Path(savePath)
SAVE_DIR.mkdir(parents=True, exist_ok=True)


seed = 42   # 국민룰 42 
random.seed(seed) 
np.random.seed(seed)
torch.manual_seed(seed)



def gradient_descent_update(w, b, dw, db, lr):
    w = w - lr * dw
    b = b - lr * db
    
    return w, b


def accuracy(w, b, inputs, label):
    net = model.BinaryClassifierGraph()
    prediction = net.forward(w, inputs, b)
    pred = prediction >= 0.5
    pred = pred.squeeze()
    label = label.type(torch.bool)
    count = len(label)
    
    correct_pred = torch.sum(torch.eq(pred, label))
    accuracy = correct_pred * 1.0/count
    return accuracy


def main(): 
    logging.info('1. Create Data points')

    num_points = 500  # sample points 
    class_zeros = torch.empty(num_points, 2).normal_(mean=2, std=0.5)   # Sampling 500-point in 2-tuple with mean=2, std=0.5
    class_ones = torch.empty(num_points, 2).normal_(mean=4, std=0.7)   # Sampling 500-point in 2-tuple with mean=2, std=0.5


    plt.scatter(class_zeros[:,0], class_zeros[:,1], s=8, color='b', label='Class:0')
    plt.scatter(class_ones[:,0], class_ones[:,1], s=8, color='r', label='Class:1')
    plt.legend()
    plt.xlim([-1, 8])
    plt.ylim([-1, 8])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig("outputs/01_create_data_points.svg")  # (ref) svg  vs. png: https://ithub.tistory.com/75
    plt.close()



    logging.info('2. Data preprocessing for PyTorch')
    
    label_zero = torch.zeros_like(class_zeros[:, 0], dtype=int)
    label_one = torch.ones_like(class_ones[:, 0], dtype=int)

    label = torch.cat([label_zero, label_one])  # 레이블 
    data_points = torch.cat([class_zeros, class_ones], dim=0)  # 입력 데이터 
    
    print(f"Data points size: {data_points.shape}") # (1000, 2) shape 
    print(f"Label size: {label.shape}")  # (1000,) shape

    

    logging.info('3. Implementing the Perceptron') 

    net = model.BinaryClassifierGraph()



    logging.info('4. Start training') 

    input_size = 2 
    w = torch.randn(input_size, 1) # initial weights 
    b = torch.zeros(1)

    epochs = 100
    lr = 0.01 
    batch_size = 10 

    avg_train_loss = np.array([])  # for storing loss of each batch 
    updated_params = [] 

    for epoch in range(epochs):

        # for storing loss of each batch in current epoch
        avg_loss = np.array([])

        num_baches = int(len(label)/batch_size)

        # Shuffle data and label
        shuffled_index = random.sample(range(len(label)), len(label))  
        s_data_points = data_points[shuffled_index]
        s_label = label[shuffled_index]        

        print(f'\nEpoch: {epoch+1}')

        for batch_idx in range(num_baches): # batch_iteration 
            # get training data in batch
            start_index = batch_idx * batch_size
            end_index = (batch_idx + 1) * batch_size

            data = s_data_points[start_index : end_index]
            g_truth = s_label[start_index : end_index]

            # forward pass 
            net.forward(w, data, b)            

            # Find loss
            loss = net.loss(g_truth)

            # Backward will find gradient using chain rule 
            net.backward()            

            # Get gradients after they are updated using the backward function
            grad_w, grad_b = net.gradients()


            # Update parameters using gradient descent
            w, b = gradient_descent_update(w, b, grad_w, grad_b, lr)


            # to show training results
            avg_loss = np.append(avg_loss, [loss])
            avg_train_loss = np.append(avg_train_loss, [loss])

            time.sleep(0.001)

            print( f"\rBatch: {batch_idx+1}/{num_baches} | Avg Batch Loss: {avg_loss.mean():.3} | Batch Loss: {loss.item():.3} | Avg Train Loss:{avg_train_loss.mean():.3}" )

        # storing parameters to show decision boundary animition
        updated_params.append((w.data[0][0].clone(), w.data[1][0].clone(), b.data[0].clone()))



    logging.info('5. Plot the Decision Boundary') 
    
    plt.scatter(class_zeros[:,0], class_zeros[:,1], s=8, color='b', label='Class:0')
    plt.scatter(class_ones[:,0], class_ones[:,1], s=8, color='r', label='Class:1')
    
    x1 = torch.linspace(-1, 8, 1000)
    x2 = -(b.data[0] + w.data[0][0] * x1)/ w.data[1][0]
    plt.plot(x1, x2, c='g', label = f'Learned decision boundary:\n{w.data[0][0]:.2}x1 + {w.data[1][0]:.2}x2 + {b.data[0]:.2} = 0')
    plt.legend()
    plt.xlim([-1, 8])
    plt.ylim([-1, 8])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig("outputs/02_plot_the_devision_boundary.svg")
    plt.close()



    logging.info('6. Plot the Loss Curve') 

    plt.plot(range(len(avg_train_loss)), avg_train_loss, color='C1')  # (ref color): https://matplotlib.org/3.1.1/tutorials/colors/colors.html
    plt.xlabel('Batch no.')
    plt.ylabel('Batch Loss')
    plt.savefig("outputs/03_plot_the_loss_curve.svg")
    plt.close()


    print(f"Accuray: {accuracy(w, b, data_points, label)}")



    logging.info('6. Decision Boundary Animation') 

    for idx , (w0, w1, b) in tqdm(enumerate(updated_params)):

        fig, ax = plt.subplots()

        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 8)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        
        ax.scatter(class_zeros[:,0], class_zeros[:,1], s=8, color='b', label='Class:0')
        ax.scatter(class_ones[:,0], class_ones[:,1], s=8, color='r', label='Class:1')

        
        x1 = torch.linspace(-1, 8, 1000)
        x2 = -(b + w0 * x1)/ w1 
        plt.plot(x1, x2, color='g', label='Decision Boundary')        
        ax.legend(loc='upper right')

        plt.savefig(f"outputs/{idx}_fitting.png")

        plt.close()





if __name__ == "__main__":

    main() 
    