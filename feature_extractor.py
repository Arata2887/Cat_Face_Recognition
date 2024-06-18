import keras


class FeatureExtractionModel(keras.models.Model):
    def __init__(self, *args, **kwargs):
        super(FeatureExtractionModel, self).__init__(*args, **kwargs)

        # 加载MobileNetV3Large作为基模型
        self.base_model = keras.applications.MobileNetV2(include_top=False,
                                                         weights='imagenet',
                                                         input_shape=(224, 224, 3))
        
        # 初始不训练基模型
        self.base_model.trainable = True
        
        self.model = keras.Sequential()

        # 定义模型的其他层
        self.model.add(self.base_model)

        self.model.add(keras.layers.GlobalAveragePooling2D())
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(512, activation='relu'))
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(256, activation='relu'))
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(9 * 2, activation='linear'))

        self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

    def unfreeze_model(self, layers_to_unfreeze=-3):
        """
        解冻基模型的顶层。如果layers_to_unfreeze是负数，它表示从顶部开始解冻的层数。
        """
        # 解冻基模型的顶层
        for layer in self.base_model.layers[layers_to_unfreeze:]:
            layer.trainable = True


class MyCNNFeatureExtractionModel(keras.models.Model):
    def __init__(self, *args, **kwargs):
        super(MyCNNFeatureExtractionModel, self).__init__(*args, **kwargs)
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(224, 224, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(256, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),

            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),

            keras.layers.Dense(9 * 2, activation='linear') # 9个关键点，每个点有x和y坐标
        ])

        self.model.summary()

    def call(self, inputs):
        return self.model(inputs)
