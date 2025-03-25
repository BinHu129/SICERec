import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim//2, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim//2, output_dim, bias=True)
        )

    def forward(self, x):
        return self.mlp(x)


class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Aggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)



class UserModel(nn.Module):
    def __init__(self, emb_dim, user_emb, item_emb, rating_emb, ranking_emb, species_emb):
        super(UserModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rating_emb = rating_emb
        self.ranking_emb = ranking_emb
        self.species_emb = species_emb

        self.g_v = MLP(2 * self.emb_dim, self.emb_dim)
        self.g_p = MLP(2 * self.emb_dim, self.emb_dim)
        self.g_a = MLP(2 * self.emb_dim, self.emb_dim)

        self.user_item_attn = MLP(2 * self.emb_dim, 1)
        self.aggr_items = Aggregator(self.emb_dim, self.emb_dim)

        self.user_user_attn = MLP(2 * self.emb_dim, 1)
        self.aggr_neighbors = Aggregator(self.emb_dim, self.emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU()
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-10

    def forward(self, uids, u_item_pad, u_itemk_pad, u_itemp_pad, u_user_pad, u_user_item_pad, u_user_itemk_pad,
                u_user_itemp_pad):
        q_a = self.item_emb(u_item_pad[:, :, 0])
        u_item_er = self.rating_emb(u_item_pad[:, :, 1])
        u_item_species = self.species_emb(u_itemp_pad[:, :, 1])
        x_iq = self.g_v(torch.cat([q_a, u_item_er], dim=2).view(-1, 2 * self.emb_dim)).view(q_a.size())
        x_ip = self.g_p(torch.cat([q_a, u_item_species], dim=2).view(-1, 2 * self.emb_dim)).view(q_a.size())
        x_ia = self.g_a(torch.cat([x_iq, x_ip], dim=2).view(-1, 2 * self.emb_dim)).view(q_a.size())
        mask_u = torch.where(u_item_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        p_i = self.user_emb(uids).unsqueeze(1).expand_as(x_ia)
        alpha = self.user_item_attn(torch.cat([x_ia, p_i], dim=2).view(-1, 2 * self.emb_dim)).view(mask_u.size())
        alpha = torch.exp(alpha) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)
        h_iI = self.aggr_items(torch.sum(alpha.unsqueeze(2).expand_as(x_ia) * x_ia, 1))
        q_a_s = self.item_emb(u_user_item_pad[:, :, :, 0])
        u_user_item_er = self.rating_emb(u_user_item_pad[:, :, :, 1])
        u_user_item_species = self.species_emb(u_user_itemp_pad[:, :, :, 1])
        x_iq_s = self.g_v(torch.cat([q_a_s, u_user_item_er], dim=2).view(-1, 2 * self.emb_dim)).view(q_a_s.size())
        x_ip_s = self.g_p(torch.cat([q_a_s, u_user_item_species], dim=2).view(-1, 2 * self.emb_dim)).view(q_a_s.size())
        x_ia_s = self.g_a(torch.cat([x_iq_s, x_ip_s], dim=2).view(-1, 2 * self.emb_dim)).view(q_a_s.size())
        mask_s = torch.where(u_user_item_pad[:, :, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        p_i_s = self.user_emb(u_user_pad).unsqueeze(2).expand_as(x_ia_s)
        alpha_s = self.user_item_attn(torch.cat([x_ia_s, p_i_s], dim=3).view(-1, 2 * self.emb_dim)).view(mask_s.size())
        alpha_s = torch.exp(alpha_s) * mask_s
        alpha_s = alpha_s / (torch.sum(alpha_s, 2).unsqueeze(2).expand_as(alpha_s) + self.eps)
        h_oI_temp = torch.sum(alpha_s.unsqueeze(3).expand_as(x_ia_s) * x_ia_s, 2)
        h_oI = self.aggr_items(h_oI_temp.view(-1, self.emb_dim)).view(h_oI_temp.size())
        beta = self.user_user_attn(torch.cat([h_oI, self.user_emb(u_user_pad)], dim=2).view(-1, 2 * self.emb_dim)).view(
            u_user_pad.size())
        mask_su = torch.where(u_user_pad > 0, torch.tensor([1.], device=self.device),
                              torch.tensor([0.], device=self.device))
        beta = torch.exp(beta) * mask_su
        beta = beta / (torch.sum(beta, 1).unsqueeze(1).expand_as(beta) + self.eps)
        h_iS = self.aggr_neighbors(torch.sum(beta.unsqueeze(2).expand_as(h_oI) * h_oI, 1))
        h_i1 = self.mlp(torch.cat([h_iI, h_iS], dim=1))
        q_a = self.item_emb(u_item_pad[:, :, 0])
        u_item_er = self.rating_emb(u_item_pad[:, :, 1])
        u_item_nk = self.ranking_emb(u_itemk_pad[:, :, 1])
        x_ia = self.g_v(torch.cat([q_a, u_item_nk], dim=2).view(-1, 2 * self.emb_dim)).view(q_a.size())
        x_ia = x_ia * u_item_er
        x_ia = self.g_v(torch.cat([q_a, x_ia], dim=2).view(-1, 2 * self.emb_dim)).view(q_a.size())
        mask_u = torch.where(u_item_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        p_i = self.user_emb(uids).unsqueeze(1).expand_as(x_ia)
        alpha = self.user_item_attn(torch.cat([x_ia, p_i], dim=2).view(-1, 2 * self.emb_dim)).view(mask_u.size())
        alpha = torch.exp(alpha) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)
        h_iI = self.aggr_items(torch.sum(alpha.unsqueeze(2).expand_as(x_ia) * x_ia, 1))
        q_a_s = self.item_emb(u_user_item_pad[:, :, :, 0])
        u_user_item_er = self.rating_emb(u_user_item_pad[:, :, :, 1])
        u_user_item_nk = self.ranking_emb(u_user_itemk_pad[:, :, :, 1])
        x_ia_s = self.g_v(torch.cat([q_a_s, u_user_item_nk], dim=2).view(-1, 2 * self.emb_dim)).view(q_a_s.size())
        x_ia_s = x_ia_s * u_user_item_er
        x_ia_s = self.g_v(torch.cat([q_a_s, x_ia_s], dim=2).view(-1, 2 * self.emb_dim)).view(q_a_s.size())
        mask_s = torch.where(u_user_item_pad[:, :, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        p_i_s = self.user_emb(u_user_pad).unsqueeze(2).expand_as(x_ia_s)
        alpha_s = self.user_item_attn(torch.cat([x_ia_s, p_i_s], dim=3).view(-1, 2 * self.emb_dim)).view(mask_s.size())
        alpha_s = torch.exp(alpha_s) * mask_s
        alpha_s = alpha_s / (torch.sum(alpha_s, 2).unsqueeze(2).expand_as(alpha_s) + self.eps)
        h_oI_temp = torch.sum(alpha_s.unsqueeze(3).expand_as(x_ia_s) * x_ia_s, 2)
        h_oI = self.aggr_items(h_oI_temp.view(-1, self.emb_dim)).view(h_oI_temp.size())
        beta = self.user_user_attn(torch.cat([h_oI, self.user_emb(u_user_pad)], dim=2).view(-1, 2 * self.emb_dim)).view(
            u_user_pad.size())
        mask_su = torch.where(u_user_pad > 0, torch.tensor([1.], device=self.device),
                              torch.tensor([0.], device=self.device))
        beta = torch.exp(beta) * mask_su
        beta = beta / (torch.sum(beta, 1).unsqueeze(1).expand_as(beta) + self.eps)
        h_iS = self.aggr_neighbors(torch.sum(beta.unsqueeze(2).expand_as(h_oI) * h_oI, 1))
        h_i2 = self.mlp(torch.cat([h_iI, h_iS], dim=1))
        h_i = self.mlp(torch.cat([h_i1, h_i2], dim=1))
        return h_i


class ItemModel(nn.Module):
    def __init__(self, emb_dim, user_emb, item_emb, rating_emb, ranking_emb, species_emb):
        super(ItemModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rating_emb = rating_emb
        self.ranking_emb = ranking_emb
        self.species_emb = species_emb

        self.g_u = MLP(2 * self.emb_dim, self.emb_dim)

        self.item_users_attn = MLP(2 * self.emb_dim, 1)

        self.aggr_users = Aggregator(self.emb_dim, self.emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU()
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eps = 1e-10

    def forward(self, iids, i_user_pad, i_userk_pad, i_user_species_pad):

        p_t = self.user_emb(i_user_pad[:, :, 0])

        i_user_er = self.rating_emb(i_user_pad[:, :, 1])

        i_user_species = self.species_emb(i_user_species_pad[:, :, 1])

        mask_i = torch.where(i_user_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))

        f_jt = self.g_u(torch.cat([p_t, i_user_species], dim=2).view(-1, 2 * self.emb_dim)).view(p_t.size())

        f_jt = f_jt * i_user_er

        q_j = self.item_emb(iids).unsqueeze(1).expand_as(f_jt)

        mu_jt = self.item_users_attn(torch.cat([f_jt, q_j], dim=2).view(-1, 2 * self.emb_dim)).view(mask_i.size())

        mu_jt = torch.exp(mu_jt) * mask_i

        mu_jt = mu_jt / (torch.sum(mu_jt, 1).unsqueeze(1).expand_as(mu_jt) + self.eps)

        z_j1 = self.aggr_users(torch.sum(mu_jt.unsqueeze(2).expand_as(f_jt) * f_jt, 1))

        p_t = self.user_emb(i_user_pad[:, :, 0])

        i_user_er = self.rating_emb(i_user_pad[:, :, 1])

        i_user_nk = self.ranking_emb(i_userk_pad[:, :, 1])

        mask_i = torch.where(i_user_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))

        f_jt = self.g_u(torch.cat([p_t, i_user_er], dim=2).view(-1, 2 * self.emb_dim)).view(p_t.size())

        f_jt = f_jt * i_user_nk

        f_jt = self.g_u(torch.cat([p_t, f_jt], dim=2).view(-1, 2 * self.emb_dim)).view(p_t.size())

        q_j = self.item_emb(iids).unsqueeze(1).expand_as(f_jt)

        mu_jt = self.item_users_attn(torch.cat([f_jt, q_j], dim=2).view(-1, 2 * self.emb_dim)).view(mask_i.size())

        mu_jt = torch.exp(mu_jt) * mask_i

        mu_jt = mu_jt / (torch.sum(mu_jt, 1).unsqueeze(1).expand_as(mu_jt) + self.eps)

        z_j2 = self.aggr_users(torch.sum(mu_jt.unsqueeze(2).expand_as(f_jt) * f_jt, 1))

        z_j = self.mlp(torch.cat([z_j1, z_j2], dim=1))
        return z_j


class GraphRec(nn.Module):
    def __init__(self, n_users, n_items, n_ratings, n_ranking, n_species, emb_dim=64):
        super(GraphRec, self).__init__()
        self.n_users = n_users

        self.n_items = n_items

        self.n_ratings = n_ratings
        self.n_ranking = n_ranking
        self.n_species = n_species

        self.emb_dim = emb_dim


        self.user_emb = nn.Embedding(self.n_users, self.emb_dim, padding_idx=0)

        self.item_emb = nn.Embedding(self.n_items, self.emb_dim, padding_idx=0)

        self.rating_emb = nn.Embedding(self.n_ratings, self.emb_dim, padding_idx=0)

        self.ranking_emb = nn.Embedding(self.n_ranking, self.emb_dim, padding_idx=0)
        self.species_emb = nn.Embedding(self.n_species, self.emb_dim, padding_idx=0)

        self.user_model = UserModel(self.emb_dim, self.user_emb, self.item_emb, self.rating_emb, self.ranking_emb,
                                    self.species_emb)

        self.item_model = ItemModel(self.emb_dim, self.user_emb, self.item_emb, self.rating_emb, self.ranking_emb,
                                    self.species_emb)

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1)
        )

    def forward(self, uids, iids, u_item_pad, u_itemk_pad, u_itemp_pad, u_user_pad, u_user_item_pad, u_user_itemk_pad,
                u_user_itemp_pad, i_user_pad, i_userk_pad, i_user_species_pad):
        h_i = self.user_model(uids, u_item_pad, u_itemk_pad, u_itemp_pad, u_user_pad, u_user_item_pad, u_user_itemk_pad,
                              u_user_itemp_pad)

        z_j = self.item_model(iids, i_user_pad, i_userk_pad, i_user_species_pad)

        r_ij = self.mlp(torch.cat([h_i, z_j], dim=1))
        return r_ij