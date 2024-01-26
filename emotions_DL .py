from taker import Model
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim

class model(torch.nn.Module):
    def __init__(self, input_size, interm_layer_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, interm_layer_size)
        self.norm1 = nn.BatchNorm1d(interm_layer_size)
        self.linear2 = torch.nn.Linear(interm_layer_size, interm_layer_size)
        self.norm2 = nn.BatchNorm1d(interm_layer_size)
        self.linear3 = torch.nn.Linear(interm_layer_size, 6)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
    def forward(self, x):
        x = self.relu1(self.norm1(self.linear1(x)))
        x = self.relu2(self.norm2(self.linear2(x)))
        x = self.linear3(x)
        return x.squeeze()  


def emotion(layer, input_size, interm_layer_size):

    net = model(input_size, interm_layer_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    lis = ["train", "validation"] #to train the model on train and validation dataset
    #lis = ["train"]
    for h in lis:

        data_train = dataset[h]
        data_X_train = data_train['text']

        train_loss = 0
        acc=0
        acc_max = 0

        print('Training time')
        net.train()
        for i in range(len(data_X_train)):
            X_train_layer = m.get_residual_stream(data_X_train[i])[layer].to(torch.float)#[n_token*288]
            n_token = X_train_layer.shape[0]
            lab = data_train['label'][i]
            tensor = torch.zeros(6)
            tensor[lab] = 1
            label = torch.tile(tensor, (n_token, 1))#[n_token*6]
            label = label.to(device).to(torch.float) 

            net.zero_grad()
            output = net(X_train_layer)#[n_token*6]
            loss = criterion(output, label)

            acc_max += n_token
            acc_pred = torch.argmax(output, dim=1) #prediction from my model [n_token]
            acc_lab = torch.full((n_token,), lab) #true labels [n_token]
            acc += (acc_pred == acc_lab.to(device)).sum().item()

            train_loss +=loss
            loss.backward()
            optimizer.step()
            if i%100 == 0 and i!=0:
                print('Epoch', i, (train_loss/100).item(), acc/acc_max)
                train_loss=0
                acc = 0
                acc_max = 0


    print('Testing time')
    data_test = dataset["test"]
    data_X_test = data_test['text']

    test_loss = 0
    total_test_loss = 0
    acc=0
    acc_max = 0

    #net.eval() #Perfs are worst in eval mode, so we stay in train mode for the test
    with torch.no_grad():
        for i in range(len(data_X_test)):
            X_test_layer = m.get_residual_stream(data_X_test[i])[layer].to(torch.float)#[n_token*288]
            n_token = X_test_layer.shape[0]
            lab = data_test['label'][i]
            tensor = torch.zeros(6)
            tensor[lab] = 1
            label = torch.tile(tensor, (n_token, 1))#[n_token*6]
            label = label.to(device).to(torch.float) 

            output = net(X_test_layer)
            loss = criterion(output, label)

            acc_max += n_token
            acc_pred = torch.argmax(output, dim=1) #prediction from my model [n_token]
            acc_lab = torch.full((n_token,), lab) #true labels [n_token]
            acc += (acc_pred == acc_lab.to(device)).sum().item()

            total_test_loss +=loss
            test_loss +=loss

            if i%100 == 0 and i!=0:
                print('Epoch', i, (test_loss/100).item())
                test_loss=0

    final = total_test_loss/len(data_X_test)
    print('Layer', layer, 'CE score:', final.item(), 'Accuracy:', acc/acc_max)
    return net


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)

mod = "nickypro/tinyllama-15M"
#mod = "nickypro/tinyllama-42M"

m = Model(mod)

if mod == "nickypro/tinyllama-15M":
    input_size, interm_layer_size = 288, 192
elif mod == "nickypro/tinyllama-42M":
    input_size, interm_layer_size = 512, 380


#for lay in range(m.cfg.n_layers*2 + 1):
lay = 2 #the best layer
net = emotion(lay, input_size, interm_layer_size)