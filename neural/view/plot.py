
import matplotlib.pyplot as plt


def plot_one(target, validation, epochs):
    plt.figure(1)
    plt.plot(epochs, target, 'b', label='mean')
    plt.plot(epochs, validation, 'g', label='validation')
    plt.title('Log Loss')
    plt.xlabel('epochs')
    plt.ylabel('log loss')
    plt.legend()
    plt.show()


def read_file():
    with open('run_log.csv', 'rt') as file_reader:
        lines = []
        for line in file_reader:
            lines.append(line.split(','))
        lines = lines[1:]
        epochs = [int(x[0]) for x in lines]
        meas = [float(x[2]) for x in lines]
        vals = [float(x[4]) for x in lines]
        plot_one(meas, vals, epochs)


if __name__ == '__main__':
    read_file()
