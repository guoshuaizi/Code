import os


##文件路径设置

#实验名称
experiment_name = "实例训练"
#训练名称
train_name = "实例训练"

test_name="实例训练"
#训练集数据文件路径
train_data = "./dataset/US"

#
train_data_2 = "./dataset/SynUS"
#验证数据集
val_path = "val_data"
#源域测试集
# test_data_1 = "predict_data104"
# #跨域测试集
# test_data_2 = "predict_data84"

test_data = "dataset/test"
#部分训练结果存储路径
save_path = "train_images"
#pth文件保存的总目录
save_param = "params"
##训练参数设置

#训练batch_size
train_batch_size = 2
#加载pth文件与否
load_model = False
#下载的权重路径
weight_path= ""
#初始学习速率
inital_learning_rate = 2e-5
#训练轮数
num_epoch = 2
##配置完毕，训练运行train.py

##test配置

#进行test的pth文件路径
prd_model = "params/五折交叉验证/one/unet_model-ep-1.pth"


value = 10