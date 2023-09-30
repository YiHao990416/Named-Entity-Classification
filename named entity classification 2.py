# Importing the libraries needed
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from transformers import  DistilBertForTokenClassification, DataCollatorForTokenClassification
import torch.nn as nn
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score

     
from torch import cuda
#Identify and use the GPU
if torch.cuda.is_available():    

    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

print(torch.version.cuda)

#Define the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Read and load json files into dataframe
df_train=pd.read_json("train.json",lines=True)
df_test=pd.read_json("test.json",lines=True)
df_valid=pd.read_json("valid.json",lines=True)

# check the format of the dataframe
print(df_train.head(5))

# Preprocess the label by adding "-100" at the beginning to refresent "CLS"
def preprocess_label(df):
    new_tags=[]
    for i in range(len(df)):
        tags=df["tags"].iloc[i]
        tags.insert(0,-100)
        new_tags.append(tags)
    df["new_tags"]=new_tags

preprocess_label(df_train)
preprocess_label(df_test)
preprocess_label(df_valid)

# Collate function from Huggingface
collate_fn=DataCollatorForTokenClassification(tokenizer=tokenizer,padding='longest' ,label_pad_token_id=-100,return_tensors="pt")

class NERdataset(Dataset):
    def __init__(self,x,y,max_len,i):
        self.x=x
        self.y=y
        self.max_len=max_len
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        x=self.x[index]
        y=self.y[index]

        encoding=tokenizer.encode_plus(
            x,
            truncation = True,
            return_tensors='pt',
        )
        return {"input_ids":encoding['input_ids'].flatten(), "attention_mask":encoding['attention_mask'].flatten(), "labels":torch.tensor(y,dtype=torch.long)}

class DistilbertNER(nn.Module):
    def __init__(self,num_labels):

        super(DistilbertNER,self).__init__()
        self.pretrained=DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased",num_labels=num_labels)
    
    def forward(self,input_ids,attention_mask, labels=None):
        if labels==None:

            out=self.pretrained(input_ids=input_ids,attention_mask=attention_mask)
            return out
        
        out=self.pretrained(input_ids=input_ids,attention_mask=attention_mask,labels=labels)

        return out
    
#Define parameter for dataset and dataloader
batch_size_train= 16
batch_size_test=16
batch_size_valid=16
max_len_train=140
max_len_test=140
max_len_valid=140
number_workers=1

dataset_train=NERdataset(df_train["tokens"],df_train["new_tags"],max_len=max_len_train,i=0)
dataset_test=NERdataset(df_test["tokens"],df_test["new_tags"],max_len=max_len_test,i=0)
dataset_valid=NERdataset(df_valid["tokens"],df_valid["new_tags"],max_len=max_len_valid,i=0)

##for data in [dataset_train, dataset_test, dataset_valid]:
##    lengths = [data[i]["labels"].shape[0] for i in range(len(data))]
##    print(lengths)
##    assert all([l<=140 for l in lengths])


dataloader_train=DataLoader(dataset_train,batch_size=batch_size_train,shuffle=True,collate_fn=collate_fn)##,num_workers = number_workers)
dataloader_test=DataLoader(dataset_test,batch_size=batch_size_test,shuffle=False,collate_fn=collate_fn)##,num_workers = number_workers)
dataloader_valid=DataLoader(dataset_valid,batch_size=batch_size_valid,shuffle=False,collate_fn=collate_fn)##,num_workers = number_workers)

#Define function used for training the model
model=DistilbertNER(5)
loss_function=torch.nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.SGD(params=model.parameters(),lr=2e-4)   #Modify to optimize the training 
total_training_samples=len(dataset_train)
total_testing_samples=len(dataset_test)
epochs= 2



def train(model,dataloader_train,dataloader_valid,optimizer,epochs):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_acc_train = 0
        total_f1_train=0
        train_iterations= len(dataset_train)/batch_size_train
        j=0
      
        for batch in dataloader_train:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            output=model(input_ids,attention_mask,labels)
            loss, logits= output.loss, output.logits

            predicted_label=logits.argmax(dim=-1)
            predicted_label = predicted_label[labels != -100]
            labels = labels[labels != -100]
            predicted_label=predicted_label.to("cpu")
            labels=labels.to("cpu")
            acc = accuracy_score(labels,predicted_label)
            f1 = f1_score(labels, predicted_label, average = "weighted")
            total_acc_train+=acc
            total_f1_train+= f1
            loss.backward()
            optimizer.step()

            j+=1
            print(f"The batch {j} train accuracy is {(acc)*100} % , F1 score={f1}")

        print(f"\nEpoch {epoch+1}:the training accuracy is {(total_acc_train/train_iterations)*100}% and the F1-score is {total_f1_train/train_iterations}\n")  
        
        k=0
        total_acc_valid = 0
        total_f1_valid=0
        valid_iterations=len(dataset_valid)/batch_size_valid
        model.eval()
        with torch.no_grad():
            for batch in dataloader_valid:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                output=model(input_ids,attention_mask,labels)
                loss, logits= output.loss, output.logits
                predicted_label=logits.argmax(dim=-1)
                predicted_label = predicted_label[labels != -100]
                labels = labels[labels != -100]
                predicted_label=predicted_label.to("cpu")
                labels=labels.to("cpu")
                acc = accuracy_score(labels,predicted_label)
                f1 = f1_score(labels, predicted_label, average = "weighted")
            
                total_acc_valid+=acc
                total_f1_valid += f1

                k+=1
                print(f"The batch {k} validating accuracy is {(acc)*100} % , F1 score={f1}")


        print(f"\nEpoch {epoch+1}:the validating ccuracy is {(total_acc_valid/valid_iterations)*100}% and the F1-score is {total_f1_valid/valid_iterations}\n")        


train(model,dataloader_train,dataloader_valid,optimizer,epochs)




    


    
