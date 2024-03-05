from matplotlib import pyplot as plt

from classes.in_out import In_Out
from classes.processing import Processing
from labs.sem2 import lab1
from labs.sem2 import lab2
from labs.sem2 import lab3
from labs.sem2 import lab4

# Для запуска лабы впишите её номер в импорт и в условный оператор и запустите проект
if __name__ == '__main__':
    # Экземпляры классов
    new_in_out = In_Out()
    new_processing = Processing()

    lab2.main()

    # img1
    img_name = 'img1'
    c_gamma = 2
    gamma = 0.40
    c_log = 10
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

