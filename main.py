#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import datetime
from Recdata import GraphDataset
from Recdata import collate_fn
from Modules import   GraphRec



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device - ' + str(device))
batch_size = 128
embed_dim = 64
learning_rate = 0.001
n_epochs = 30


with open('data/trained models epinions/dataset_epinions.pkl', 'rb') as f:
    train_set = pickle.load(f)
    valid_set = pickle.load(f)
    test_set = pickle.load(f)

with open('data/trained models epinions/list_epinions.pkl', 'rb') as f:
    u_items_list = pickle.load(f)
    u_itemk_list= pickle.load(f)
    u_itemp_list= pickle.load(f)
    
    u_users_list = pickle.load(f)
    
    u_users_items_list = pickle.load(f)
    u_users_itemk_list= pickle.load(f)
    u_users_itemp_list= pickle.load(f)
    
    i_users_list = pickle.load(f)
    i_userk_list= pickle.load(f)
    i_users_species_list=pickle.load(f)


    (user_count ,item_count ,rate_count ,species_count,pagerank_count) = pickle.load(f)



train_data = GraphDataset(train_set, u_items_list,u_itemk_list,u_itemp_list, u_users_list, u_users_items_list,u_users_itemk_list, u_users_itemp_list,i_users_list,i_userk_list, i_users_species_list)

valid_data = GraphDataset(valid_set, u_items_list,u_itemk_list,u_itemp_list, u_users_list, u_users_items_list,u_users_itemk_list, u_users_itemp_list,i_users_list,i_userk_list, i_users_species_list)

test_data = GraphDataset(test_set,u_items_list,u_itemk_list,u_itemp_list, u_users_list, u_users_items_list, u_users_itemk_list,u_users_itemp_list,i_users_list,i_userk_list, i_users_species_list)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)

model = GraphRec(user_count+1, item_count+1, rate_count+1,pagerank_count+1,species_count+1, embed_dim).to(device)

optimizer = torch.optim.RMSprop(model.parameters(), learning_rate)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.1,last_epoch=-1)

best_rmse = 9999.0
best_mae = 9999.0
endure_count = 0


for epoch in range(n_epochs):

    # Training step
    model.train()
    #模型训练
    s_loss = 0
    #误差
    
    for i, (uids, iids,species,labels,u_items,u_itemk,u_itemp, u_users, u_users_items,u_users_itemk,u_users_itemp,i_users,i_userk,i_users_species) in tqdm(enumerate(train_loader), total=len(train_loader)):
        uids = uids.to(device)
        iids = iids.to(device)
        labels = labels.to(device)
        species=species.to(device)
        u_items = u_items.to(device)
        u_users = u_users.to(device)
        u_users_items = u_users_items.to(device)
        i_users = i_users.to(device)
        i_users_species=i_users_species.to(device)
        u_itemp=u_itemp.to(device)
        u_users_itemp=u_users_itemp.to(device)
        u_itemk=u_itemk.to(device)
        u_users_itemk=u_users_itemk.to(device)
        i_userk=i_userk.to(device)
        
        optimizer.zero_grad()
        outputs = model(uids, iids, u_items,u_itemk,u_itemp, u_users, u_users_items,u_users_itemk,u_users_itemp,i_users, i_userk,i_users_species)
        loss = criterion(outputs, labels.unsqueeze(1))

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        s_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

    # Validate step
    model.eval()
    #验证集选择合适参数
    errors = []
    with torch.no_grad():
        for uids, iids,species,labels,u_items,u_itemk,u_itemp, u_users, u_users_items,u_users_itemk,u_users_itemp,i_users,i_userk,i_users_species in tqdm(valid_loader):
            uids = uids.to(device)
            iids = iids.to(device)
            labels = labels.to(device)
            species=species.to(device)
            u_items = u_items.to(device)
            u_users = u_users.to(device)
            u_users_items = u_users_items.to(device)
            i_users = i_users.to(device)
            i_users_species=i_users_species.to(device)
            u_itemp=u_itemp.to(device)
            u_users_itemp=u_users_itemp.to(device)
            u_itemk=u_itemk.to(device)
            u_users_itemk=u_users_itemk.to(device)
            i_userk=i_userk.to(device)

            preds = model(uids, iids, u_items,u_itemk,u_itemp, u_users, u_users_items,u_users_itemk,u_users_itemp,i_users, i_userk,i_users_species)
            error = torch.abs(preds.squeeze(1) - labels)
            errors.extend(error.data.cpu().numpy().tolist())
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.power(errors, 2)))

    scheduler.step()

    ckpt_dict = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    #模型信息保存{训练epoch数，Graph参数，optimizer优化器参数保存}
    torch.save(ckpt_dict, 'data/trained models epinions/latest_checkpoint.pth')

    if best_mae > mae:
        best_mae = mae
        best_rmse = rmse
        endure_count = 0
        torch.save(ckpt_dict, 'data/trained models epinions/best_checkpoint_{}.pth'.format(embed_dim))
    else:
        endure_count += 1

    print('Epoch {} validation: MAE: {:.4f}, RMSE: {:.4f}, Best MAE: {:.4f}'.format(epoch+1, mae, rmse, best_mae))
    if endure_count > 5:
        torch.save(ckpt_dict, 'data/trained models epinions/final_checkpoint_{}.pth'.format(embed_dim))
        print('finished')
                # 获取当前日期和时间
        current_datetime = datetime.datetime.now()

        # 打印当前日期和时间
        print("当前日期和时间：", current_datetime)
        break

embed_dim = 64
checkpoint = torch.load('data/trained models epinions/best_checkpoint_{}.pth'.format(embed_dim))
model = GraphRec(user_count+1, item_count+1, rate_count+1,pagerank_count+1,species_count+1,embed_dim).to(device)
model.load_state_dict(checkpoint['state_dict'])

model.eval()
test_errors = []
with torch.no_grad():
    for uids, iids,species,labels,u_items,u_itemk,u_itemp, u_users, u_users_items,u_users_itemk,u_users_itemp,i_users,i_userk,i_users_species in tqdm(test_loader):
        uids = uids.to(device)
        iids = iids.to(device)
        labels = labels.to(device)
        species=species.to(device)
        u_items = u_items.to(device)
        u_users = u_users.to(device)
        u_users_items = u_users_items.to(device)
        i_users = i_users.to(device)
        i_users_species=i_users_species.to(device)
        u_itemp=u_itemp.to(device)
        u_users_itemp=u_users_itemp.to(device)
        u_itemk=u_itemk.to(device)
        u_users_itemk=u_users_itemk.to(device)
        i_userk=i_userk.to(device)

        
        preds = model(uids, iids, u_items,u_itemk,u_itemp, u_users, u_users_items,u_users_itemk,u_users_itemp,i_users, i_userk,i_users_species)
        error = torch.abs(preds.squeeze(1) - labels)
        test_errors.extend(error.data.cpu().numpy().tolist())

test_mae = np.mean(test_errors)
test_rmse = np.sqrt(np.mean(np.power(test_errors, 2)))
print('Test: MAE: {:.4f}, RMSE: {:.4f}'.format(test_mae, test_rmse))






