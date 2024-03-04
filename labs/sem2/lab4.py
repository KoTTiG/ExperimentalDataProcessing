import matplotlib.pyplot as plt
import numpy as np

from classes.in_out import In_Out
from classes.processing import Processing


def main():
    # Экземпляры классов
    new_in_out = In_Out()
    new_processing = Processing()

    # grace
    grace_file_name = 'grace'
    img_grace = new_in_out.read_jpg(grace_file_name)
    neg_grace = new_processing.negative(img_grace, 255)
    new_in_out.write_jpg(neg_grace, grace_file_name + '_negative')

    # xcr 1
    xcr_1_file_name = 'data/xcr/c12-85v.xcr'
    xcr_1_shape = (1024, 1024)

    xcr_1_data = new_in_out.read_xcr(xcr_1_file_name, xcr_1_shape)
    xcr_1_data_recount = xcr_1_data
    neg_xcr_1 = new_processing.negative(xcr_1_data, 255)


    new_in_out.write_jpg(neg_xcr_1, xcr_1_file_name + '_negative')

    # xcr 2
    xcr_2_file_name = 'data/xcr/u0.xcr'
    xcr_2_shape = (2500, 2048)
    xcr_2_data = new_in_out.read_xcr(xcr_2_file_name, xcr_2_shape)
    xcr_2_data_recount = xcr_2_data
    neg_xcr_2 = new_processing.negative(xcr_2_data, 255)

    # Negative plots
    plt.subplot(321)
    new_in_out.show_jpg_sub(img_grace, grace_file_name + ' original')
    plt.subplot(322)
    new_in_out.show_jpg_sub(neg_grace, grace_file_name + ' negative')

    plt.subplot(323)
    new_in_out.show_jpg_sub(xcr_1_data_recount, xcr_1_file_name + ' original')
    plt.subplot(324)
    new_in_out.show_jpg_sub(neg_xcr_1, xcr_1_file_name + ' negative')
    plt.subplot(325)
    new_in_out.show_jpg_sub(xcr_2_data_recount, xcr_2_file_name + ' original')
    plt.subplot(326)
    new_in_out.show_jpg_sub(neg_xcr_2, xcr_2_file_name + ' negative')

    plt.show()

    # img1
    img_name = 'img1'
    c_gamma = 2
    gamma = 1.10
    c_log = 30
    img_data = new_in_out.read_jpg(img_name)

    img_gamma = new_processing.gamma_transform(img_data, c_gamma, gamma)
    img_log = new_processing.log_transform(img_data, c_log)
    # save to files
    new_in_out.write_jpg(img_gamma, img_name + '_gamma')
    new_in_out.write_jpg(img_log, img_name + '_log')

    # plot img
    plt.subplot(221)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name), 'original')
    plt.subplot(222)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name + '_gamma'),
                            'gamma transform, gamma=' + str(gamma) + ', C=' + str(c_gamma))
    plt.subplot(223)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name + '_log'), 'log transform, C=' + str(c_log))
    plt.show()

    # img2
    img_name = 'img2'
    img_data = new_in_out.read_jpg(img_name)

    img_gamma = new_processing.gamma_transform(img_data, c_gamma, gamma)
    img_log = new_processing.log_transform(img_data, c_log)
    # save to files
    new_in_out.write_jpg(img_gamma, img_name + '_gamma')
    new_in_out.write_jpg(img_log, img_name + '_log')

    # plot img
    plt.subplot(221)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name), 'original')
    plt.subplot(222)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name + '_gamma'),
                            'gamma transform, gamma=' + str(gamma) + ', C=' + str(c_gamma))
    plt.subplot(223)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name + '_log'), 'log transform, C=' + str(c_log))
    plt.show()

    # img3
    img_name = 'img3'
    c_gamma = 5
    gamma = 0.67
    c_log = 20
    img_data = new_in_out.read_jpg(img_name)

    img_gamma = new_processing.gamma_transform(img_data, c_gamma, gamma)
    img_log = new_processing.log_transform(img_data, c_log)
    # save to files
    new_in_out.write_jpg(img_gamma, img_name + '_gamma')
    new_in_out.write_jpg(img_log, img_name + '_log')

    # plot img
    plt.subplot(221)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name), 'original')
    plt.subplot(222)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name + '_gamma'),
                            'gamma transform, gamma=' + str(gamma) + ', C=' + str(c_gamma))
    plt.subplot(223)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name + '_log'), 'log transform, C=' + str(c_log))
    plt.show()

    # img4
    img_name = 'img4'
    c_gamma = 5
    gamma = 0.4
    c_log = 5
    img_data = new_in_out.read_jpg(img_name)

    img_gamma = new_processing.gamma_transform(img_data, c_gamma, gamma)
    img_log = new_processing.log_transform(img_data, c_log)
    # save to files
    new_in_out.write_jpg(img_gamma, img_name + '_gamma')
    new_in_out.write_jpg(img_log, img_name + '_log')

    # plot img
    plt.subplot(221)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name), 'original')
    plt.subplot(222)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name + '_gamma'),
                            'gamma transform, gamma=' + str(gamma) + ', C=' + str(c_gamma))
    plt.subplot(223)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name + '_log'), 'log transform, C=' + str(c_log))
    plt.show()

    # img4
    img_name = 'HollywoodLC'
    c_gamma = 3
    gamma = 0.7
    c_log = 6
    img_data = new_in_out.read_jpg(img_name)

    img_gamma = new_processing.gamma_transform(img_data, c_gamma, gamma)
    img_log = new_processing.log_transform(img_data, c_log)
    # save to files
    new_in_out.write_jpg(img_gamma, img_name + '_gamma')
    new_in_out.write_jpg(img_log, img_name + '_log')

    # plot img
    plt.subplot(221)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name), 'original')
    plt.subplot(222)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name + '_gamma'),
                            'gamma transform, gamma=' + str(gamma) + ', C=' + str(c_gamma))
    plt.subplot(223)
    new_in_out.show_jpg_sub(new_in_out.read_jpg(img_name + '_log'), 'log transform, C=' + str(c_log))
    plt.show()
