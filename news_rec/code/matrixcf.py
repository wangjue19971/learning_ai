import tensorflow as tf
import os

class MCF(object):

    def __init__(self):
        # 初始化模型参数
        self.feature_name_list = ['user', 'item']
        self.label_name = 'ctr'
        self.feature_size = 40000
        self.embedding_size = 16
        self.lr = 0.01
        self.checkpoint_dir = "news_rec/model/matrixcf"
        self.checkpoint_interval = True

        self.file_dir = "news_rec/data"
        self.n_threads = 2
        self.shuffle = 50
        self.batch = 16

        self.epochs = 10
        self.current_model_path = "news_rec/model/save_model"
        self.restore = False

    def init_model(self):
        # 定义模型结构
        user = tf.keras.Input(shape=(1,), name='user', dtype=tf.int64)
        item = tf.keras.Input(shape=(1,), name='item', dtype=tf.int64)
        
        embed = tf.keras.layers.Embedding(self.feature_size, self.embedding_size)
        
        user_emb = embed(user)
        item_emb = embed(item)
        
        # 合并用户和物品的嵌入向量
        interact = tf.keras.layers.Multiply()([user_emb, item_emb])
        
        # 添加Dropout层防止过拟合
        # dropout_layer = tf.keras.layers.Dropout(0.2)(interact, training=True)
        
        logit = tf.reduce_sum(interact, axis=-1)
        pred = tf.keras.layers.Activation('sigmoid', name="pred")(logit)
        
        self.model = self.model = tf.keras.Model({
            "user": user,
            "item": item
        }, {
            "pred": logit,
            "user_emb": tf.identity(user_emb, name="user_emb"),
            "item_emb": tf.identity(item_emb, name="item_emb")
        })
        self.model.summary()

    def init_loss(self):
        # 定义损失函数
        self.loss = tf.keras.losses.BinaryCrossentropy()

    def init_opt(self):
        # 定义优化器
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def init_metric(self):
        # 定义评估指标
        self.metric_auc = tf.keras.metrics.AUC(name="auc")
        self.metric_loss = tf.keras.metrics.Mean(name="loss")

    def init_save_checkpoint(self):
        # 初始化模型保存机制
        checkpoint = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
        self.manager = tf.train.CheckpointManager(checkpoint, directory=self.checkpoint_dir, max_to_keep=2)

    def init_dataset(self, file_dir, is_train=True):
        # 准备输入数据
        def _parse_example(example):
            feature_description = {
                "ctr": tf.io.FixedLenFeature([], dtype=tf.float32),
                "user": tf.io.FixedLenFeature([], dtype=tf.int64),
                "item": tf.io.FixedLenFeature([], dtype=tf.int64),
            }
            return tf.io.parse_single_example(example, feature_description)

        files = tf.data.Dataset.list_files(os.path.join(file_dir, '*.tfrecord'))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=self.n_threads)
        
        if is_train:
            dataset = dataset.shuffle(self.shuffle)
        
        dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def train_step(self, ds):
        # 训练步骤
        for inputs in ds:
            with tf.GradientTape() as tape:
                target = inputs.pop(self.label_name)
                outputs = self.model(inputs, training=True)
                logits = outputs['pred']  # 修改这里，使用字典键访问预测值
                loss = self.loss(target, logits)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            self.metric_loss.update_state(loss)
            self.metric_auc.update_state(target, logits)
        
        result = {self.metric_auc.name: self.metric_auc.result().numpy(), self.metric_loss.name: self.metric_loss.result().numpy()}
        self.metric_auc.reset_states()
        self.metric_loss.reset_states()
        return result

    def eval_step(self, ds):
        # 评估步骤
        for inputs in ds:
            target = inputs.pop(self.label_name)
            outputs = self.model(inputs, training=False)
            logits = outputs['pred']  # 修改这里，使用字典键访问预测值
            self.metric_auc.update_state(target, logits)
        
        result = {self.metric_auc.name: self.metric_auc.result().numpy()}
        self.metric_auc.reset_states()
        return result

    def run(self, train_ds, test_ds, mode="train_and_eval"):
        # 运行模型
        if self.restore:
            self.manager.restore_or_initialize()
        if mode == "train_and_eval":
            for epoch in range(self.epochs):
                print(f"Epoch {epoch+1}/{self.epochs}")
                train_result = self.train_step(train_ds)
                print(f"Training result: {train_result}")
                eval_result = self.eval_step(test_ds)
                print(f"Evaluation result: {eval_result}")
                if train_result['auc'] > 0.5 or eval_result['auc'] > 0.5:
                    self.manager.save(checkpoint_number=epoch)
        elif mode == "train":
            for epoch in range(self.epochs):
                print(f"Epoch {epoch+1}/{self.epochs}")
                train_result = self.train_step(train_ds)
                print(f"Training result: {train_result}")
                if train_result['auc'] > 0.5:
                    self.manager.save(checkpoint_number=epoch)
        elif mode == "eval":
            eval_result = self.eval_step(test_ds)
            print(f"Evaluation result: {eval_result}")

    def export_model(self):
        # 导出模型
        tf.keras.models.save_model(self.model, self.current_model_path)

    def load_model(self):
        # 加载模型
        self.imported = tf.keras.models.load_model(self.current_model_path)

    def infer(self, x):
        # 推理
        return self.imported(x)

if __name__ == "__main__":
    mcf = MCF()
    train_ds = mcf.init_dataset("news_rec/data/train")
    test_ds = mcf.init_dataset("news_rec/data/test", is_train=False)
    mcf.init_model()
    mcf.init_loss()
    mcf.init_opt()
    mcf.init_metric()
    mcf.init_save_checkpoint()
    mcf.run(train_ds, test_ds, mode="train_and_eval")
    mcf.export_model()
    
# Training result: {'auc': 0.83746296, 'loss': 0.50003207}
# Evaluation result: {'auc': 0.70692934}