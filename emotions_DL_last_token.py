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

    batch_size = 5
    n_classes = 6

    lis = ["train", "validation"] #to train the model on train and validation dataset
    #lis = ["train"]
    for h in lis:

        data_train = dataset[h]
        data_X_train = data_train['text']

        train_loss = 0
        acc=0

        print('Training time on', h, 'dataset')
        net.train()
        for i in range(0, len(data_X_train)-batch_size, batch_size):
            X_train_layer = torch.stack([m.get_residual_stream(data_X_train[i])[layer,-1] for i in range(batch_size)]).to(torch.float)#[batch*288]
            
            lab = data_train['label'][i:i+batch_size] # [batch_size]
            tensor_list = []
            for j in range(batch_size):
                tensor = torch.zeros(n_classes)
                x = lab[j]
                tensor[x] = 1 #one hot vector of 6
                tensor_list.append(tensor)    
            label = torch.stack(tensor_list).to(device).to(torch.float) #[batch_size*6]

            net.zero_grad()
            output = net(X_train_layer)#[batch*6]
            loss = criterion(output, label)

            acc_pred = torch.argmax(output, dim =-1) #prediction from my model [batch]
            label_pred = torch.argmax(label, dim =-1) #true labels [batch]
            acc += torch.sum(torch.eq(label_pred, acc_pred)).item()

            train_loss +=loss
            loss.backward()
            optimizer.step()
            if i%100 == 0 and i!=0:
                print('Epoch', i, (train_loss/100).item(), acc/100)
                train_loss=0
                acc = 0


    print('Testing time')
    data_test = dataset["test"]
    data_X_test = data_test['text']

    test_loss = 0
    total_test_loss = 0
    acc=0

    #net.eval() #Perfs are worst in eval mode, so we stay in train mode for the test
    with torch.no_grad():
        for i in range(0, len(data_X_test), batch_size):
            X_test_layer = torch.stack([m.get_residual_stream(data_X_test[i])[layer,-1] for i in range(batch_size)]).to(torch.float)#[batch*288]
            
            lab = data_test['label'][i:i+batch_size] # [batch_size]
            for j in range(batch_size):
                tensor = torch.zeros(n_classes)
                x = lab[j]
                tensor[x] = 1 #one hot vector of 6
                tensor_list.append(tensor)
            label = torch.stack(tensor_list).to(device).to(torch.float) #[batch_size*6] 

            output = net(X_test_layer)
            loss = criterion(output, label)
            total_test_loss +=loss
            test_loss +=loss

            if i%100 == 0 and i!=0:
                print('Epoch', i, (test_loss/100).item())
                test_loss=0

    final = total_test_loss/len(data_X_test)
    print('Layer', layer, 'CE score:', final.item())
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
