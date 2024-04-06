import os
import torch
import warnings
import torch
import torch.nn as nn
import visdom
import torch.optim as optim
from torch.onnx import export
from sklearn.metrics import precision_recall_curve, auc
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dataloader import cr_dataloader, cr_dataloaderbase
from model import MyModel
from mymetric import mymetric

warnings.filterwarnings("ignore")

# 注意：不报错或者不爆显存的话记得把下边这一行注掉，否则会肉眼可见的拖慢效率
# 经常爆显存，cuda核心断言报错，为了方便debug设环境变量同步运行，训练时应并行加速
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
# 参数定义
weights_of_distance=0.05
weights_of_position=0.3
weight_of_feature=2
weights_of_pronoun=0.05
batch_size=8
lr=0.01
gamma=0.1
step_size=50
epochs=100
early_stopping=20
# 数据预处理
train_data,validation_data,test_data,samp_single,feature_df=cr_dataloaderbase()
# 生成dataloader
train_loader=cr_dataloader(train_data,weights_of_distance,weights_of_position,weight_of_feature,weights_of_pronoun,samp_single,feature_df,batch_size)
valid_loader=cr_dataloader(validation_data,weights_of_distance,weights_of_position,weight_of_feature,weights_of_pronoun,samp_single,feature_df,batch_size)
test_loader=cr_dataloader(test_data,weights_of_distance,weights_of_position,weight_of_feature,weights_of_pronoun,samp_single,feature_df,batch_size)
# 设置训练硬件
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型实例化并传入GPU
model=MyModel().to(device)
# 损失
criterion = nn.BCELoss()
# 优化器
optimizer = optim.Adam(model.parameters(), lr)
# 学习率动态衰减
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
stop=0
pre_total_loss=1e9
# 开始训练
for epoch in range(epochs):
    model.train()
    total_loss=0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Train")):
        # print(target.shape)
        data, target = data.to(device), target.to(device)
        # 排除可能的nan值影响（此处偶发报错未找到原因，怀疑内存和显存太小）
        data[torch.isnan(data)]=0.0
        # 求预测值，其实输出已经是一维了，加squeeze是确保安全
        output = model(data).squeeze()
        # 默认label是0或1的int，转换成float才能运算
        target = torch.tensor(target, dtype=torch.float32)
        # 保证损失计算二者矩阵形状相同
        # output=output.view(batch_size,-1)
        target=target.view_as(output)
        # MSE求损失
        loss = criterion(output, target)
        # 加loss.item而不是loss是因为loss自带计算图，会导致计算量暴增
        total_loss += loss.item()
        # 梯度清零
        optimizer.zero_grad()
        # 方向传播
        loss.backward()
        # 权重更新
        optimizer.step()
        # 把用完的两个释放一下显存，减小爆显存风险
        del output,loss
        torch.cuda.empty_cache()
    # 学习率动态更新
    scheduler.step()
    # 验证
    model.eval()
    val_correct = 0
    val_total = 0
    TP=0
    FN=0
    FP=0
    all_targets_v = []
    all_predictions_v = []
    # 不更新不求导
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(valid_loader, desc="valid")):
            data, target = data.to(device), target.to(device)
            data[torch.isnan(data)] = 0.0
            output = model(data).squeeze()
            target = torch.tensor(target, dtype=torch.float32)
            # output = output.view(batch_size, -1)
            target = target.view_as(output)
            # 变成float以和target进行比较
            predictions = (output > 0.5).float()
            all_targets_v.extend(target.cpu().numpy())
            all_predictions_v.extend(predictions.cpu().numpy())
            # 正确数
            val_correct += (predictions == target).sum().item()
            # 真正例
            TP+=((predictions==1.0)&(target==1.0)).sum().item()
            # 假反例
            FN+=((predictions==0.0)&(target==1.0)).sum().item()
            # 假正例
            FP+=((predictions==1.0)&(target==0.0)).sum().item()
            # target.size(0)是第一维宽度，也就是batch_size
            val_total += target.size(0)
    val_accuracy = val_correct / val_total
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    F1_Score=2*precision*recall/(precision+recall)
    mymetric(all_targets_v, all_predictions_v)
    print(
        f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_loader):.4f}, Recall: {recall:.4f}, Precision:{precision:.4f},Val Accuracy: {val_accuracy:.4f},F1_score: {F1_Score:.4f}, Learning Rate: {scheduler.get_lr()[0]:.6f}")
    # early_stop 该任务基本用不到
    if total_loss > pre_total_loss:
        stop += 1
        if stop > 20:
            print("长时间不进步，训练结束")
            break
        pre_total_loss=total_loss
        total_loss=0
    else:
        stop=0
print("训练完成！")

# 在测试集上评估模型
model.eval()
test_correct = 0
test_total = 0
TP_t=0
FN_t=0
FP_t=0
all_targets = []
all_predictions = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Test")):
        data, target = data.to(device), target.to(device)
        data[torch.isnan(data)] = 0.0
        output = model(data).squeeze()
        # output = output.view(batch_size, -1)
        target = torch.tensor(target, dtype=torch.float32)
        predictions = (torch.sigmoid(output) > 0.5).float()
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        # 真正例
        TP_t += ((predictions == 1.0) & (target == 1.0)).sum().item()
        # 假反例 始终为0，bug
        FN_t += ((predictions == 0.0) & (target == 1.0)).sum().item()
        # 假正例
        FP_t += ((predictions == 1.0) & (target == 0.0)).sum().item()
        test_correct += (predictions == target).sum().item()
        test_total += target.size(0)
test_accuracy = test_correct / test_total
Recall=TP_t/(TP_t+FN_t)
Precision=TP_t/(TP_t+FP_t)
F1_Score=2*Precision*Recall/(Precision+Recall)
mymetric(all_targets, all_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}, Recall: {Recall:.4f},Precision:{Precision:.4f},F1_score:{F1_Score:.4f}")
exit()