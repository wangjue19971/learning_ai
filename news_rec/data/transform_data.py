import tensorflow as tf
import random
# step01

file_path = "news_rec/data/mini_news.dat"
train_path = "news_rec/data/mini_news.train.tfrecord"
test_path = "news_rec/data/mini_news.test.tfrecord"
index_path = "news_rec/data/index"

lines = open(file_path, encoding="utf8")
train_writer = tf.io.TFRecordWriter(train_path)
test_writer = tf.io.TFRecordWriter(test_path)
index_writer = open(index_path, "w")

feature_name_list = ['ts', 'user', 'item', 'ctr']
slot = {}


def write_index(slot: dict, writer):
    for k, v in slot.items():
        writer.write(f"{k}\t{v}\n")


def to_tfrecord(line, writer):
    sample = {}
    user = line[1]
    item = line[2]
    ctr = float(line[3])
    sample["ctr"] = tf.train.Feature(float_list=tf.train.FloatList(value=[ctr]))
    # user
    if user not in slot.keys():
        slot[user] = len(slot)
    value = [slot[user]]
    sample["user"] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=value))

    # item
    if item not in slot.keys():
        slot[item] = len(slot)
    value = [slot[item]]
    sample["item"] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=value))

    sample = tf.train.Example(features=tf.train.Features(feature=sample))
    writer.write(sample.SerializeToString())


# slot -> dict
for line in lines:
    line = line.strip().split("\t")
    if random.randint(1, 10) > 8:
        to_tfrecord(line, test_writer)
    else:
        to_tfrecord(line, train_writer)

write_index(slot, index_writer)

index_writer.close()
train_writer.close()
test_writer.close()

# 初始化两个集合来存储user和item
users = set()
items = set()

# 重新遍历lines来填充集合
for line in open(file_path, encoding="utf8"):
    line = line.strip().split("\t")
    user, item = line[1], line[2]
    users.add(user)
    items.add(item)

# 打印user和item的数量
print(f"Number of unique users: {len(users)}")
print(f"Number of unique items: {len(items)}")