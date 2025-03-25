import torch
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, data, u_items_list, u_itemk_list, u_item_species, u_user_list, u_users_items_list,
                 u_users_itemk_list, u_users_item_species, i_users_list, i_userk_list, i_users_species_list):
        self.data = data
        self.u_items_list = u_items_list
        self.u_item_species = u_item_species
        self.u_itemk_list = u_itemk_list

        self.u_users_list = u_user_list

        self.u_users_items_list = u_users_items_list
        self.u_users_item_species = u_users_item_species
        self.u_users_itemk_list = u_users_itemk_list

        self.i_users_list = i_users_list
        self.i_users_species_list = i_users_species_list
        self.i_userk_list = i_userk_list

    def __getitem__(self, index):
        uid = self.data[index][0]
        iid = self.data[index][1]
        species = self.data[index][2]
        label = self.data[index][3]

        u_items = self.u_items_list[uid]
        u_itemk = self.u_itemk_list[uid]
        u_itemp = self.u_item_species[uid]

        u_users = self.u_users_list[uid]

        u_users_items = self.u_users_items_list[uid]
        u_users_itemk = self.u_users_itemk_list[uid]
        u_users_itemp = self.u_users_item_species[uid]

        i_users = self.i_users_list[iid]
        i_userk = self.i_userk_list[iid]
        i_users_species = self.i_users_species_list[iid]

        # 使训练集验证集和测试集数据信息和调入的用户和项目信息统一调入统一

        return (uid, iid, species,
                label), u_items, u_itemk, u_itemp, u_users, u_users_items, u_users_itemk, u_users_itemp, i_users, i_userk, i_users_species

    def __len__(self):
        return len(self.data)


def collate_fn(batch_data):
    uids, iids, labels, speciesp = [], [], [], []
    u_items, u_users, u_users_items, i_users, i_users_species = [], [], [], [], []
    u_itemk, u_users_itemk, i_userk = [], [], []
    u_itemp, u_users_itemp = [], []

    for data, u_items_u, u_itemk_u, u_itemp_u, u_users_u, u_users_items_u, u_users_itemk_u, u_users_itemp_u, i_users_i, i_userk_i, i_users_species_i in batch_data:

        (uid, iid, species, label) = data
        uids.append(uid)
        iids.append(iid)
        speciesp.append(species)
        labels.append(label)

        u_items.append(u_items_u)
        u_itemk.append(u_itemk_u)
        u_itemp.append(u_itemp_u)
        u_users.append(u_users_u)

        u_u_items = []
        for uui in u_users_items_u:
            u_u_items.append(uui)
        u_users_items.append(u_u_items)
        u_u_itemk = []
        for uuk in u_users_itemk_u:
            u_u_itemk.append(uuk)
        u_users_itemk.append(u_u_itemk)
        u_u_itemp = []
        for uuk in u_users_itemp_u:
            u_u_itemp.append(uuk)
        u_users_itemp.append(u_u_itemp)

        i_users.append(i_users_i)
        i_userk.append(i_userk_i)
        i_users_species.append(i_users_species_i)

    batch_size = len(batch_data)

    # padding
    u_items_maxlen = 45
    u_users_maxlen = 45
    i_users_maxlen = 45
    u_itemp_maxlen = 45
    u_itemk_maxlen = 45
    i_userk_maxlen = 45

    u_item_pad = torch.zeros([batch_size, u_items_maxlen, 2], dtype=torch.long)
    for i, ui in enumerate(u_items):
        u_item_pad[i, :len(ui), :] = torch.LongTensor(ui)
        # u_item_pad[用户数，项目数，评分]

    u_itemp_pad = torch.zeros([batch_size, u_itemp_maxlen, 2], dtype=torch.long)
    for i, up in enumerate(u_itemp):
        u_itemp_pad[i, :len(up), :] = torch.LongTensor(up)
        # u_item_pad[用户数，项目数，评分]
    u_itemk_pad = torch.zeros([batch_size, u_itemk_maxlen, 2], dtype=torch.long)
    for i, ui in enumerate(u_itemk):
        u_itemk_pad[i, :len(ui), :] = torch.LongTensor(ui)

    u_user_pad = torch.zeros([batch_size, u_users_maxlen], dtype=torch.long)
    for i, uu in enumerate(u_users):
        u_user_pad[i, :len(uu)] = torch.LongTensor(uu)
        # u_user_pad[用户数，朋友数]

    u_user_item_pad = torch.zeros([batch_size, u_users_maxlen, u_items_maxlen, 2], dtype=torch.long)
    for i, uu_items in enumerate(u_users_items):
        for j, ui in enumerate(uu_items):
            u_user_item_pad[i, j, :len(ui), :] = torch.LongTensor(ui)
            # u_user_item_pad[用户，用户朋友，用户朋友交互的项目，用户朋友交互项目的评分]
    u_user_itemk_pad = torch.zeros([batch_size, u_users_maxlen, u_itemk_maxlen, 2], dtype=torch.long)
    for i, uu_items in enumerate(u_users_items):
        for j, ui in enumerate(uu_items):
            u_user_itemk_pad[i, j, :len(ui), :] = torch.LongTensor(ui)

    u_user_itemp_pad = torch.zeros([batch_size, u_users_maxlen, u_itemp_maxlen, 2], dtype=torch.long)
    for i, uu_items in enumerate(u_users_itemp):
        for j, ui in enumerate(uu_items):
            u_user_itemp_pad[i, j, :len(ui), :] = torch.LongTensor(ui)
            # u_user_item_pad[用户，用户朋友，用户朋友交互的项目，用户朋友交互项目的评分]

    i_user_pad = torch.zeros([batch_size, i_users_maxlen, 2], dtype=torch.long)
    for i, iu in enumerate(i_users):
        i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)
        # i_user_pad[项目数，与项目交互的用户数，与项目交互的用户的评分]
    i_userk_pad = torch.zeros([batch_size, i_userk_maxlen, 2], dtype=torch.long)
    for i, iu in enumerate(i_userk):
        i_userk_pad[i, :len(iu), :] = torch.LongTensor(iu)

    i_user_species_pad = torch.zeros([batch_size, i_users_maxlen, 2], dtype=torch.long)
    for i, iu in enumerate(i_users_species):
        i_user_species_pad[i, :len(iu), :] = torch.LongTensor(iu)
        # i_user_pad[项目数，与项目交互的用户数，与项目交互的用户的评分]

    uids = torch.LongTensor(uids)
    # 用户ID
    iids = torch.LongTensor(iids)
    # 项目ID
    labels = torch.FloatTensor(labels)
    # 评分标签
    speciesp = torch.FloatTensor(speciesp)

    return uids, iids, speciesp, labels, u_item_pad, u_itemk_pad, u_itemp_pad, u_user_pad, u_user_item_pad, u_user_itemk_pad, u_user_itemp_pad, i_user_pad, i_userk_pad, i_user_species_pad

