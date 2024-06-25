
import os
import tensorflow as tf

class MatrixCF:
    def __init__(self):
        # 初始化参数
        self.feature_list = ['user', 'item']
        self.label = 'ctr'
        self.feature_size = 2365
        self.embedding_size = 64
        self.lr = 0.01
        self.batch_size = 128
        self.epoch = 50
        self.n_thread = 2
        self.shuffle = 50
        
        # 路径设置
        self.model_path = "news_rec_system/model"
        self.data_path = "news_rec_system/data"
        self.train_path = os.path.join(self.data_path, "news.train.tfrecord")
        self.test_path = os.path.join(self.data_path, "news.test.tfrecord")
        
    def init_model(self):
        # 定义模型输入
        user = tf.keras.Input(shape=(1,), name='user', dtype=tf.int64)
        item = tf.keras.Input(shape=(1,), name='item', dtype=tf.int64)
        
        # 定义Embedding层
        embed = tf.keras.layers.Embedding(input_dim=self.feature_size, output_dim=self.embedding_size, embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        
        # 获取用户和物品的嵌入向量
        user_emb = embed(user)
        item_emb = embed(item)
        
        # 计算用户和物品嵌入向量的点积
        logit = tf.keras.layers.Dot(axes=-1)([user_emb, item_emb])
        
        # 修改部分：添加Flatten层来确保输出形状与标签匹配，并使用sigmoid激活函数
        flatten = tf.keras.layers.Flatten()(logit)
        output = tf.keras.layers.Activation('sigmoid')(flatten)
        
        # 构建模型
        self.model = tf.keras.Model(inputs={"user": user, "item": item}, outputs=output)
        self.model.summary()
        
        # 编译模型
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.lr),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        
        # 模型保存路径检查
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        # 加载和预处理数据
        train_dataset = self.load_and_preprocess_data(self.train_path)
        test_dataset = self.load_and_preprocess_data(self.test_path)
        
        
        # 训练模型
        self.model.fit(train_dataset,
                       epochs=self.epoch,
                       validation_data=test_dataset,
                       verbose=1)
        
        # 保存模型
        self.model.save(os.path.join(self.model_path, "matrixcf.h5"))
        
        # 模型评估
        print("开始评估模型性能...")
        self.model.evaluate(test_dataset)
        
    def load_and_preprocess_data(self, filepath):
        # 加载TFRecord数据
        dataset = tf.data.TFRecordDataset(filepath)
        
        # 数据预处理
        def parse_tfrecord_fn(example_proto):
            feature_description = {
                'user': tf.io.FixedLenFeature([], tf.int64),
                'item': tf.io.FixedLenFeature([], tf.int64),
                'ctr': tf.io.FixedLenFeature([], tf.float32)
            }
            example = tf.io.parse_single_example(example_proto, feature_description)
            return {"user": example['user'], "item": example['item']}, example['ctr']
        
        dataset = dataset.map(parse_tfrecord_fn) 
        dataset = dataset.shuffle(self.shuffle).batch(self.batch_size)
        return dataset

if __name__ == "__main__":
    matrixcf = MatrixCF()
    matrixcf.init_model()