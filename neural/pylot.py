# testing for keras applications in python
# from keras.applications import VGG16
from keras.applications import ResNet50
# from keras.applications import InceptionV3
# from keras.applications import Xception
from keras.utils import plot_model


def view_model(top, no_top):
    idx = 0
    fmt = '{: <30} {: <30} {: <60}'
    print(fmt.format('with top', 'without top', 'class type'))
    print(fmt.format('---- ---', '------- ---', '----- ----'))
    for a, b in zip(top.layers, no_top.layers):
        print(fmt.format(str(a.name), str(b.name), str(type(a))))
        idx += 1
    for a in top.layers[idx:]:
        print(fmt.format(a.name, '', str(type(a))))
    plot_model(top, to_file='pylot.resnet.model.png', show_shapes=True, show_layer_names=True)


def view_vgg16():
    # view_model(VGG16(),
    #            VGG16(include_top=False))

    model = VGG16()
    for layer in model.layers:
        try:
            print(layer.name, layer.activation)
        except:
            pass


def view_resnet50():
    view_model(ResNet50(),
               ResNet50(include_top=False))


def view_inceptionV3():
    view_model(InceptionV3(),
               InceptionV3(include_top=False))


def view_xception():
    view_model(Xception(),
               Xception(include_top=False))


if __name__ == '__main__':
    view_resnet50()
    # view_vgg16()
    # view_vgg16()
    # view_xception()
    # with open('read.txt', 'rt') as f_obj:
    #     next(f_obj)
    #     for line in f_obj:
    #         print(line)

