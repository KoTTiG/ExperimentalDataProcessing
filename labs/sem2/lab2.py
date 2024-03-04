from matplotlib import pyplot as plt

from classes.in_out import In_Out
from classes.processing import Processing


def main():
    # Экземпляры классов
    new_in_out = In_Out()
    new_processing = Processing()

    # Данные с .xcr файла
    # file_name = 'data/xcr/c12-85v.xcr'
    # shape = (1024, 1024)
    # file_name = 'u0'
    # shape = (2500, 2048)
    # file_data = new_in_out.read_xcr(file_name, shape)
    # file_data_recount = new_processing.recount_2d(file_data, 255)
    # new_in_out.show_jpg(file_data_recount, False, 'xray')
    # new_in_out.write_jpg(file_data_recount, file_name)
    # new_in_out.write_xcr(file_data_recount, 'x-ray_' + file_name)

    file_name = 'grace'

    img = new_in_out.read_jpg(file_name)
    img = new_processing.recount_2d(img, 255)
    name = ""
    new_in_out.show_jpg(img, 0, name)
    plt.show()


    file_name = 'data/xcr/c12-85v.xcr'
    shape = (1024, 1024)
    img = new_in_out.read_xcr(file_name, shape)
    img = new_processing.recount_2d(img, 255)
    name = ""
    new_in_out.show_xray( img, name)
    plt.show()