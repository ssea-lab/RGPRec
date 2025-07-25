import torch
import random

truncate_len = 10

def collate_fn(batch_data):
    uids, iids, labels, u_avgs, i_avgs = [], [], [], [], []
    u_items_div, u_items, u_users, u_users_items, i_users = [], [], [], [], []
    u_itemsdiv_len, u_items_len, u_users_len, i_users_len = [], [], [], []

    for data, u_itemsdiv_u, u_avg_u, u_users_u, u_users_items_u, i_users_i, i_avg_i in batch_data:

        (uid, iid, label) = data
        uids.append(uid)
        iids.append(iid)
        labels.append(label)
        u_avgs.append(u_avg_u)
        i_avgs.append(i_avg_i)

        if len(u_itemsdiv_u) <= truncate_len:
            u_items_div.append(u_itemsdiv_u)
        else:
            u_items_div.append(random.sample(u_itemsdiv_u, truncate_len))
        u_itemsdiv_len.append(min(len(u_itemsdiv_u), truncate_len))

        if len(u_users_u) <= truncate_len:
            u_users.append(u_users_u)
            u_u_items = []
            for uui in u_users_items_u:
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
            u_users_items.append(u_u_items)
        else:
            sample_index = random.sample(list(range(len(u_users_u))), truncate_len)
            u_users.append([u_users_u[si] for si in sample_index])

            u_users_items_u_tr = [u_users_items_u[si] for si in sample_index]
            u_u_items = []
            for uui in u_users_items_u_tr:
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
            u_users_items.append(u_u_items)

        u_users_len.append(min(len(u_users_u), truncate_len))

        if len(i_users_i) <= truncate_len:
            i_users.append(i_users_i)
        else:
            i_users.append(random.sample(i_users_i, truncate_len))
        i_users_len.append(min(len(i_users_i), truncate_len))


    batch_size = len(batch_data)

    u_itemsdiv_maxlen = max(u_itemsdiv_len)
    u_users_maxlen = max(u_users_len)
    i_users_maxlen = max(i_users_len)

    u_itemdiv_pad = torch.zeros([batch_size, u_itemsdiv_maxlen, 2], dtype=torch.long)
    for i, uid in enumerate(u_items_div):
        u_itemdiv_pad[i, :len(uid), :] = torch.LongTensor(uid)

    
    u_user_pad = torch.zeros([batch_size, u_users_maxlen, 2], dtype=torch.long)
    for i, uu in enumerate(u_users):
        u_user_pad[i, :len(uu), :] = torch.LongTensor(uu)
    
    u_user_item_pad = torch.zeros([batch_size, u_users_maxlen, truncate_len, 2], dtype=torch.long)
    for i, uu_items in enumerate(u_users_items):
        for j, ui in enumerate(uu_items):
            u_user_item_pad[i, j, :len(ui), :] = torch.LongTensor(ui)

    i_user_pad = torch.zeros([batch_size, i_users_maxlen, 2], dtype=torch.long)
    for i, iu in enumerate(i_users):
        i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)



    return torch.LongTensor(uids), torch.LongTensor(iids), torch.FloatTensor(labels),\
            u_itemdiv_pad, torch.FloatTensor(u_avgs), u_user_pad, u_user_item_pad, i_user_pad, torch.FloatTensor(i_avgs)
