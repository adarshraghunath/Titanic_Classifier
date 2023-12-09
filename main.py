from utils import *
from model import TitanicModel
from tqdm import tqdm

#LOAD OPTIONS (JSON)

with open('options.json') as f:
    json_data = json.load(f)

gen_specs   = json_data['Features']
model_specs = json_data['Model']   #model specific
other_specs = json_data['Save']

#LOAD DATA

train_file = pd.read_csv('./train.csv')
test_file = pd.read_csv('./test.csv')

total_data = pd.concat([train_file,test_file],axis=0) #Concatenate for efficient feature engineering
total_data.reset_index(inplace=True)

ignore_features = ["index","PassengerId"]
ignore_features.extend(gen_specs["Ignore"])

total_data = total_data.drop(columns=ignore_features)   #Drop unnecessary features

cat_features = gen_specs["categorical_features"]
num_features = gen_specs["numerical_features"]


#If Name variable is considered, append 'Title' to categorical_features
if not "Name" in ignore_features:
    cat_features.append("Title")
    total_data = exec_name(total_data)  #Name variable now ready for one hot encoding


# Transform the data to fit the model
transform = Transforms(categorical_features=cat_features,numerical_features=num_features)

y_total = total_data["Survived"]
X_total = total_data.drop(columns=['Survived'])

X_total = transform(X_total)

#TRAIN DATA
X = X_total[:891]
y = y_total[:891]

#TEST DATA
X_test = X_total[891:]


#LOAD DATA 

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=gen_specs["tt_split"],random_state=35)

train_dataset = CustomDataset(X_train,  labels= y_train, transform=transform)
val_dataset = CustomDataset(X_val, labels = y_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=gen_specs["train_batch_size"], shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=gen_specs["val_batch_size"], shuffle=False)

#LOAD MODEL

input_size = X_train.shape[1]
print(X_train.shape)
model = TitanicModel(input_size=input_size, **model_specs)

#Train Loop

loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=gen_specs["LR"])


epochs = gen_specs["epochs"]
# Move the training progress bar outside the loop
train_progress_bar = tqdm(total=len(train_loader)*epochs, desc='Training', leave=False)

for epoch in range(epochs):
    model.train()

    # train_iterator = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}', leave=False)

    for item, train_set in enumerate(train_loader):
        optimizer.zero_grad()
        feature,label = train_set['features'],train_set['labels']
        output = model(feature)
        l = loss(output.squeeze(),label)

        l.backward()
        optimizer.step()

        train_progress_bar.update(1)
    

    if (epoch+1) % 100 == 0:
        model.eval()
        avg_val = 0.0
        # val_iterator = tqdm(val_loader, desc=f'Val Epoch {epoch + 1}', leave=False)
        for item, val_set in enumerate(val_loader):
            v_feature,v_label = val_set['features'],val_set['labels']
            val_output = model(v_feature)
            v_l = loss(val_output.squeeze(),v_label)
            avg_val += v_l.item()
        
        avg_val_loss = avg_val / len(val_loader)
        print(f'Val epoch : {epoch+1}, val loss : {avg_val_loss}')

train_progress_bar.close()

test_dataset = CustomDataset(X_test,training=False)
test_loader = DataLoader(test_dataset,shuffle=False)

predictions = [] 
model.eval()
for i,item in enumerate(test_loader):
    op = model(item['features'])
    prob = torch.sigmoid(op)  #Calc probabilities
    # print(prob)
    binary_pred = (prob>0.5).int()
    predictions.append(binary_pred)

predictions = torch.cat(predictions,dim=0)

if other_specs["plot"]:
    save_to_csv(test_file=test_file,predictions=predictions,**other_specs)



