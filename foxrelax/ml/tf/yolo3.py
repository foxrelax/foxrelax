from functools import wraps
import numpy as np
import tensorflow as tf
from tensorflow import keras


@wraps(keras.layers.Conv2D)
def darknet53_conv2d(*args, **kwargs):
    """
    Conv2D Wrapper
    """
    darknet_conv_kwargs = {
        'kernel_initializer': keras.initializers.random_normal(stddev=0.02)
    }
    darknet_conv_kwargs['padding'] = 'same'
    darknet_conv_kwargs.update(kwargs)
    return keras.layers.Conv2D(*args, **darknet_conv_kwargs)


def darknet53_conv2d_block(*args, **kwargs):
    """
    Conv2D + BN + LeakyReLU
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return keras.Sequential([
        darknet53_conv2d(*args, **no_bias_kwargs),
        keras.layers.BatchNormalization(axis=-1),
        keras.layers.LeakyReLU(alpha=0.1)
    ])


def darknet53_resblock(x, filters, num_blocks):
    x = darknet53_conv2d_block(filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = darknet53_conv2d_block(filters // 2, (1, 1))(x)
        y = darknet53_conv2d_block(filters, (3, 3))(y)
        x = keras.layers.Add()([x, y])
    return x


def darknet53_body(x):
    """
    DarkNet53
    输入为416x416x3的RGB Tensor, 输出为其对应的3个特征层

    x的形状变化:
    (batch_size, 416, 416, 3)
    (batch_size, 416, 416, 32)
    (batch_size, 208, 208, 64)
    (batch_size, 104, 104, 128)
    (batch_size, 52, 52, 256)
    (batch_size, 26, 26, 512)
    (batch_size, 13, 13, 1024)

    >>> x = tf.random.normal((1, 416, 416, 3))
    >>> feat1, feat2, feat3 = darknet53_body(x)
    >>> assert feat1.shape == (1, 52, 52, 256)
    >>> assert feat2.shape == (1, 26, 26, 512)
    >>> assert feat3.shape == (1, 13, 13, 1024)

    参数:
    x的形状: (batch_size, 416, 416, 3)

    返回: (feat1, feat2, feat3)
    feat1的形状: (batch_size, 52, 52, 256)
    feat2的形状: (batch_size, 26, 26, 512)
    feat3的形状: (batch_size, 13, 13, 1024)
    """
    # (batch_size, 416, 416, 3) -> (batch_size, 416, 416, 32)
    x = darknet53_conv2d_block(32, (3, 3))(x)
    # (batch_size, 416, 416, 32) -> (batch_size, 208, 208, 64)
    x = darknet53_resblock(x, 64, 1)
    # (batch_size, 208, 208, 64) -> (batch_size, 104, 104, 128)
    x = darknet53_resblock(x, 128, 2)
    # (batch_size, 104, 104, 128) -> (batch_size, 52, 52, 256)
    x = darknet53_resblock(x, 256, 8)
    feat1 = x
    # (batch_size, 52, 52, 256) -> (batch_size, 26, 26, 512)
    x = darknet53_resblock(x, 512, 8)
    feat2 = x
    # (batch_size, 26, 26, 512) -> (batch_size, 13, 13, 1024)
    x = darknet53_resblock(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3


def make_five_conv(x, filters):
    x = darknet53_conv2d_block(filters, (1, 1))(x)
    x = darknet53_conv2d_block(filters * 2, (3, 3))(x)
    x = darknet53_conv2d_block(filters, (1, 1))(x)
    x = darknet53_conv2d_block(filters * 2, (3, 3))(x)
    x = darknet53_conv2d_block(filters, (1, 1))(x)
    return x


def make_yolo_head(x, filters, out_filters):
    y = darknet53_conv2d_block(filters * 2, (3, 3))(x)
    y = darknet53_conv2d(out_filters, (1, 1))(y)
    return y


def yolo_body(input_shape, anchors_mask, num_classes):
    """
    YOLO V3 FPN(Feature Pyramid Networks)

    >>> anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    >>> num_classes = 80
    >>> x = tf.random.normal((1, 416, 416, 3))
    >>> [p5, p4, p3] = yolo_body((416, 416, 3), anchors_mask, num_classes)(x)
    >>> assert p5.shape == (1, 13, 13, 255)
    >>> assert p4.shape == (1, 26, 26, 255)
    >>> assert p3.shape == (1, 52, 52, 255)

    返回: [P5, P4, P3]
    P5的形状: (batch_size, 13, 13, 3*(num_classes+5))
    P4的形状: (batch_size, 26, 26, 3*(num_classes+5))
    P3的形状: (batch_size, 52, 52, 3*(num_classes+5))
    """
    inputs = keras.Input(input_shape)
    # C3的形状: (batch_size, 52, 52, 256)
    # C4的形状: (batch_size, 26, 26, 512)
    # C5的形状: (batch_size, 13, 13, 1024)
    C3, C4, C5 = darknet53_body(inputs)

    # 第一个特征层
    # x的形状:
    # (batch_size, 13, 13, 1024) ->
    # (batch_size, 13, 13, 512) ->
    # (batch_size, 13, 13, 1024) ->
    # (batch_size, 13, 13, 512) ->
    # (batch_size, 13, 13, 1024) ->
    # (batch_size, 13, 13, 512)
    x = make_five_conv(C5, 512)
    # P5的形状: (batch_size, 13, 13, 3*(num_classes+5))
    P5 = make_yolo_head(x, 512, len(anchors_mask[0]) * (num_classes + 5))

    # x的形状: (batch_size, 13, 13, 512) -> (batch_size, 13, 13, 256)
    x = darknet53_conv2d_block(256, (1, 1))(x)
    # x的形状: (batch_size, 13, 13, 256) -> (batch_size, 26, 26, 256)
    x = keras.layers.UpSampling2D(2)(x)
    # x的形状: (batch_size, 26, 26, 256) + (batch_size, 26, 26, 512) ->
    # (batch_size, 26, 26, 768)
    x = keras.layers.Concatenate(axis=-1)([x, C4])

    # 第二个特征层
    # x的形状:
    # (batch_size, 26, 26, 768) ->
    # (batch_size, 26, 26, 256) ->
    # (batch_size, 26, 26, 512) ->
    # (batch_size, 26, 26, 256) ->
    # (batch_size, 26, 26, 512) ->
    # (batch_size, 26, 26, 256)
    x = make_five_conv(x, 256)
    # P4的形状: (batch_size, 26, 26, 3*(num_classes+5))
    P4 = make_yolo_head(x, 256, len(anchors_mask[1]) * (num_classes + 5))

    # x的形状: (batch_size, 26, 26, 256) -> (batch_size, 26, 26, 128)
    x = darknet53_conv2d_block(128, (1, 1))(x)
    # x的形状: (batch_size, 26, 26, 128) -> (batch_size, 52, 52, 128)
    x = keras.layers.UpSampling2D(2)(x)
    # x的形状: (batch_size, 52, 52, 128) + (batch_size, 52, 52, 256) ->
    # (batch_size, 52, 52, 384)
    x = keras.layers.Concatenate()([x, C3])

    # 第三个特征层
    # x的形状:
    # (batch_size, 52, 52, 384) ->
    # (batch_size, 52, 52, 128) ->
    # (batch_size, 52, 52, 256) ->
    # (batch_size, 52, 52, 128) ->
    # (batch_size, 52, 52, 256) ->
    # (batch_size, 52, 52, 128)
    x = make_five_conv(x, 128)
    # P3的形状: (batch_size, 52, 52, 3*(num_classes+5))
    P3 = make_yolo_head(x, 128, len(anchors_mask[2]) * (num_classes + 5))
    return keras.Model(inputs, [P5, P4, P3])


def get_classes(dataset='voc'):
    if dataset == 'voc':
        return get_voc_classes()


def get_voc_classes():
    class_names = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    return class_names, len(class_names)


def get_anchors():
    anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                        [59, 119], [116, 90], [156, 198], [373, 326]])
    return anchors, len(anchors)


def get_anchors_and_decode(feats,
                           anchors,
                           num_classes,
                           input_shape,
                           calc_loss=False):
    """
    >>> feats = tf.random.normal((1, 13, 13, 255))
    >>> anchors = tf.constant([[116, 90], [156, 198], [373, 326]])
    >>> num_classes = 80
    >>> input_shape = (416, 416)
    >>> box_xy, box_wh, box_confidence, box_class_probs = get_anchors_and_decode(
            feats, anchors, num_classes, input_shape, calc_loss=False)
    >>> assert box_xy.shape == (1, 13, 13, 3, 2)
    >>> assert box_wh.shape == (1, 13, 13, 3, 2)
    >>> assert box_confidence.shape == (1, 13, 13, 3, 1)
    >>> assert box_class_probs.shape == (1, 13, 13, 3, 80)
    
    >>> grid, feats, box_xy, box_wh = get_anchors_and_decode(feats,
                                                             anchors,
                                                             num_classes,
                                                             input_shape,
                                                             calc_loss=True)
    >>> assert grid.shape == (13, 13, 3, 2)
    >>> assert feats.shape == (1, 13, 13, 3, 5 + 80)
    >>> assert box_xy.shape == (1, 13, 13, 3, 2)
    >>> assert box_wh.shape == (1, 13, 13, 3, 2)

    参数:
    feats的形状: (batch_size, 52, 52, 255=3*(num_classes+5))
                (batch_size, 26, 26, 255=3*(num_classes+5))
                (batch_size, 13, 13, 255=3*(num_classes+5))
    anchors的形状: (3, 2)
    num_classes: 标量
    input_shape的形状: (2,), 内容=(416, 416)
    calc_loss: boolean

    返回:
    假设feats.shape=(batch_size, 13, 13, 255), num_anchors=3, num_classes=80
    
    <1> calc_loss=True时(grid, feats, box_xy, box_wh)
        grid的形状: (13, 13, 3, 2)
        feats的形状: (batch_size, 13, 13, 3, 5 + 80)
        box_xy的形状: (batch_size, 13, 13, 3, 2)
        box_wh的形状: (batch_size, 13, 13, 3, 2)

    <2> calc_loss=False时(box_xy, box_wh, box_confidence, box_class_probs)
        box_xy的形状: (batch_size, 13, 13, 3, 2)
        box_wh的形状: (batch_size, 13, 13, 3, 2)
        box_confidence的形状: (batch_size, 13, 13, 3, 1)
        box_class_probs的形状: (batch_size, 13, 13, 3, 80)
    """
    # 假设:
    # 以feats.shape=(batch_size, 13, 13, 255), num_anchors=3, num_classes=80来描述整个流程的形状变化

    num_anchors = len(anchors)
    # grid_shape的形状: (2, ), 内容=(13,13)
    grid_shape = tf.shape(feats)[1:3]
    # grid_x的形状: (13, 13, num_anchors=3, 1)
    # grid_y的形状: (13, 13, num_anchors=3, 1)
    # grid的形状: (13, 13, num_anchors=3, 2)
    #
    # 最左上角的网格左上角的坐标(除以13就可以归一化):
    # grid[0][0] = [[0. 0.]
    #               [0. 0.]
    #               [0. 0.]]
    # 最右下角的网格左上角的坐标(除以13就可以归一化):
    # grid[12][12] = [[12. 12.]
    #                 [12. 12.]
    #                 [12. 12.]]
    grid_x = tf.tile(tf.reshape(tf.range(grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, num_anchors, 1])
    grid_y = tf.tile(tf.reshape(tf.range(grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], num_anchors, 1])
    grid = tf.cast(tf.concat([grid_x, grid_y], axis=-1), feats.dtype)

    # anchors_tensor的形状: (1, 1, num_anchors=3, 2)
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, num_anchors, 2])
    # anchors_tensor的形状: (13, 13, num_anchors=3, 2)
    anchors_tensor = tf.cast(
        tf.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1]),
        feats.dtype)

    # feats的形状: (batch_size, 13, 13, 255=3*(5+80)) ->
    # (batch_size, 13, 13, num_anchors=3, 5+80)
    # 最后一个维度: 80表示每个类的置信度
    #             5可以拆分成4+1, 4代表的是中心宽高的调整参数, 1表示框的置信度
    feats = tf.reshape(
        feats,
        [-1, grid_shape[0], grid_shape[1], num_anchors, 5 + num_classes])

    # 注意: x,y,w,h取值在[0-1]之间
    # box_xy的形状: (batch_size, 13, 13, 3, 2)
    # box_wh的形状: (batch_size, 13, 13, 3, 2)
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(
        grid_shape[::-1], feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(
        input_shape[::-1], feats.dtype)

    # box_confidence的形状: (batch_size, 13, 13, 3, 1)
    # box_class_probs的形状: (batch_size, 13, 13, 3, num_classes=80)
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    if calc_loss == True:
        # grid的形状: (13, 13, num_anchors=3, 2)
        # feats的形状:  (batch_size, 13, 13, num_anchors=3, 5+80)
        # box_xy的形状: (batch_size, 13, 13, 3, 2) - 数据在0-1之间
        # box_wh的形状: (batch_size, 13, 13, 3, 2) - 数据在0-1之间
        return grid, feats, box_xy, box_wh

    # box_xy的形状: (batch_size, 13, 13, 3, 2) - 数据在0-1之间
    # box_wh的形状: (batch_size, 13, 13, 3, 2) - 数据在0-1之间
    # box_confidence的形状: (batch_size, 13, 13, 3, 1) - 数据在0-1之间
    # box_class_probs的形状: (batch_size, 13, 13, 3, num_classes=80) - 数据在0-1之间
    return box_xy, box_wh, box_confidence, box_class_probs