import numpy as np


class Points():

    def __init__(self, params, cnt):
        self.params = params
        self.cd = np.zeros([cnt, 3])  # coordinate
        self.dens = np.zeros(cnt)  # density
        self.Ncd = 0  # num of effective coordinates
        self.ori_grid = np.zeros([cnt, 3])  # grid index
        self.ori_dens = np.zeros(cnt)
        self.Nori = 0  # num of effective grid points

    def recover_density(self):
        self.dens = self.dens * self.Nori

    # def Merge_point(self, mrc, point_path):
    #     dcut = self.params['m'] / mrc.widthx  # distance cut off
    #     d2cut = dcut**2
    #     rdcut = self.params['f']
    #     if self.Ncd == 0:
    #         print('not running merge shifting parts')
    #         self.merged_cd_dens = np.array([])
    #         self.merged_data = np.array([])
    #         return
    #     # we mark i with j when point i is merged to point j
    #     self.member = np.arange(self.Ncd)  # label for each points
    #     self.stock = np.ones(self.Ncd)
    #     # find the dmin and dmax in
    #     dmin = np.min(self.dens)
    #     dmax = np.max(self.dens)
    #     drange = dmax - dmin
    #     print(f'here we get the density range {drange}')
    #     rv_range = 1.0 / drange
    #     tmp_Ncd = int(self.Ncd)
    #     tmp_dens = np.array(self.dens)
    #     tmp_stock = np.array(self.stock)
    #     tmp_cd = np.array(self.cd)
    #     tmp_member = np.array(self.member)
    #     self.stock, self.member = acc_merge_point(tmp_Ncd, tmp_dens, dmin,
    #                                               rv_range, rdcut, tmp_stock,
    #                                               tmp_cd, d2cut, tmp_member)
