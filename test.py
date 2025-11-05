# chemneural-network
# finding new lib material with neutral network
# 注释版本
# 导入numpy库，用于数值计算和数组操作
import numpy as np
# 导入pandas库，用于数据处理和Excel文件读取
import pandas as pd
# 导入PyTorch库，用于构建和训练神经网络
import torch
# 导入PyTorch的nn模块，包含神经网络层和损失函数等
import torch.nn as nn
# 从torch.optim导入Adam优化器，用于模型参数优化
from torch.optim import Adam
# 从torch.optim.lr_scheduler导入CosineAnnealingLR，用于学习率调度
from torch.optim.lr_scheduler import CosineAnnealingLR
# 从torch.utils.data导入DataLoader和TensorDataset，用于数据加载和批处理
from torch.utils.data import DataLoader, TensorDataset
# 导入warnings库，用于处理警告信息
import warnings
# 忽略所有警告信息，避免输出干扰
warnings.filterwarnings("ignore")


# ---------------------- 1. 回归神经网络定义（与文档S12一致，复用PLD网络结构） ----------------------
# 定义一个用于属性回归的神经网络类，继承自nn.Module
class PropertyRegressionNet(nn.Module):
    # 初始化方法，定义网络结构
    def __init__(self):
        # 调用父类nn.Module的初始化方法
        super(PropertyRegressionNet, self).__init__()
        # 定义特征提取器：由线性层、层归一化和ReLU激活函数组成的序列
        # 功能：将1维输入（SSA数值）转换为16维特征（符合文档S12要求的维度）
        self.feature_extractor = nn.Sequential(
            # 线性层：输入特征数1，输出特征数16
            nn.Linear(in_features=1, out_features=16),
            # 层归一化：对16维特征进行归一化，稳定训练过程，避免梯度消失问题
            nn.LayerNorm(normalized_shape=16),
            # ReLU激活函数：引入非线性变换，帮助模型捕捉SSA排序中的复杂关系
            nn.ReLU()
        )
        # 定义回归头：由线性层和Sigmoid激活函数组成的序列
        # 功能：将16维特征转换为1维输出（适配[0,1]范围的排序标签）
        self.regressor = nn.Sequential(
            # 线性层：输入特征数16，输出特征数1
            nn.Linear(in_features=16, out_features=1),
            # Sigmoid激活函数：将输出压缩到[0,1]范围，与排序标签范围一致
            nn.Sigmoid()
        )
    
    # 前向传播方法，定义数据在网络中的流动过程
    def forward(self, x):
        # 将输入x传入特征提取器，得到16维特征（形状为[batch_size, 16]），即SSA的16维初步表征
        feature = self.feature_extractor(x)
        # 将16维特征传入回归头，得到1维输出（形状为[batch_size, 1]），即SSA排序标签的预测值
        output = self.regressor(feature)
        # 返回预测值和16维特征
        return output, feature


# ---------------------- 2. 数据预处理（SSA专项适配：m²/g单位+数据清洗） ----------------------
# 定义SSA数据预处理函数
def preprocess_ssa_data(excel_path, ssa_col_name="SSA", ssa_unit="m²/g"):
    """
    输入：
        excel_path: SSA数据的Excel文件路径
        ssa_col_name: Excel中SSA列的标题（默认"SSA"）
        ssa_unit: SSA单位（固定为m²/g，文档S11标准）
    输出：训练/验证/测试集DataLoader、全量SSA输入Tensor（用于提取表征）
    """
    # 步骤1：读取Excel文件（支持.xlsx格式，需安装openpyxl引擎）
    try:
        # 使用pandas读取Excel文件，指定引擎为openpyxl
        df = pd.read_excel(excel_path, engine="openpyxl")
    # 捕获文件未找到错误
    except FileNotFoundError:
        # 抛出值错误，提示文件路径问题
        raise ValueError(f"Excel文件未找到，请检查路径：{excel_path}")
    # 捕获其他读取错误
    except Exception as e:
        # 抛出运行时错误，提示文件读取失败的可能原因
        raise RuntimeError(f"Excel读取失败：{str(e)}（建议检查文件是否损坏或格式是否为.xlsx）")
    
    # 步骤2：提取SSA列并校验列名是否存在
    if ssa_col_name not in df.columns:
        # 若列名不存在，抛出值错误，提示用户确认列标题
        raise ValueError(f"Excel中未找到'{ssa_col_name}'列，请确认列标题与数据匹配（例：'SSA(m²/g)'需对应修改参数）")
    # 提取SSA列的数值，得到原始SSA数据（单位：m²/g）
    raw_ssa = df[ssa_col_name].values
    # 记录初始样本数量
    initial_samples = len(raw_ssa)
    # 打印读取结果信息
    print(f"读取Excel完成：初始样本数={initial_samples}，SSA单位={ssa_unit}")

    # 步骤3：SSA专项数据清洗（严格遵循文档S11要求）
    # 3.1 剔除空值（NaN）
    raw_ssa = raw_ssa[~np.isnan(raw_ssa)]
    # 3.2 剔除0值（文档S11规定：SSA=0的样本无效，无实际比表面积）
    raw_ssa = raw_ssa[raw_ssa != 0]
    # 3.3 剔除异常值（参考CoRE MOF数据库：SSA通常在10~6000 m²/g，超出此范围视为异常）
    raw_ssa = raw_ssa[(raw_ssa >= 10) & (raw_ssa <= 6000)]
    # 记录清洗后的有效样本数量
    valid_samples = len(raw_ssa)

    # 校验有效样本数是否为0
    if valid_samples == 0:
        # 若有效样本数为0，抛出值错误，提示用户检查数据
        raise ValueError(f"清洗后无有效SSA样本！请检查Excel数据：\n- 单位是否为{ssa_unit}\n- 数值是否在10~6000{ssa_unit}范围内（正常MOF的SSA范围）")
    # 打印数据清洗结果
    print(f"数据清洗完成：有效样本数={valid_samples}（剔除空值/0值/异常值）")

    # 步骤4：生成排序标签（文档S12核心：学习SSA的相对排序而非绝对数值）
    # 先对raw_ssa进行两次排序，得到每个样本的排名（从1开始）
    # 第一次argsort得到升序索引，第二次argsort将索引转换为排名，+1确保排名从1开始
    rank = np.argsort(np.argsort(raw_ssa)) + 1
    # 将排名归一化到[0,1]范围（适配Sigmoid输出范围），公式为（排名-1）/(最大排名-1)
    rank_norm = (rank - 1) / (valid_samples - 1)

    # 步骤5：转换为Tensor并扩展维度（模型输入需为[样本数, 1]的二维张量）
    # 将raw_ssa转换为float32类型的Tensor，并在第1维增加一个维度，形状变为[valid_samples, 1]
    ssa_input = torch.tensor(raw_ssa, dtype=torch.float32).unsqueeze(1)
    # 将归一化后的排名转换为float32类型的Tensor，并扩展维度，形状变为[valid_samples, 1]
    ssa_label = torch.tensor(rank_norm, dtype=torch.float32).unsqueeze(1)

    # 步骤6：划分训练/验证/测试集（比例7:2:1，确保数据分布一致）
    # 计算各数据集的样本数量
    train_size = int(0.7 * valid_samples)
    val_size = int(0.2 * valid_samples)
    test_size = valid_samples - train_size - val_size

    # 按计算的大小分割输入数据和标签
    train_x, val_x, test_x = torch.split(ssa_input, [train_size, val_size, test_size])
    train_y, val_y, test_y = torch.split(ssa_label, [train_size, val_size, test_size])

    # 步骤7：构建DataLoader（文档S12指定batch_size=64）
    # 训练集：使用TensorDataset包装输入和标签，设置batch_size=64，打乱数据（shuffle=True）
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    # 验证集：不打乱数据（shuffle=False），便于结果复现
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64, shuffle=False)
    # 测试集：不打乱数据
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=64, shuffle=False)

    # 打印数据集划分结果
    print(f"数据集划分：训练集={train_size} | 验证集={val_size} | 测试集={test_size}")
    # 返回各数据集的DataLoader和全量SSA输入Tensor
    return train_loader, val_loader, test_loader, ssa_input


# ---------------------- 3. 模型训练（参数严格匹配文档S12，SSA与PLD共用训练逻辑） ----------------------
# 定义SSA模型训练函数
def train_ssa_model(train_loader, val_loader, device, save_model_path):
    # 初始化模型，并将其移动到指定计算设备（CPU或GPU）
    model = PropertyRegressionNet().to(device)
    # 定义损失函数：文档S12指定使用MSE损失（均方误差），适用于回归任务，适配排序标签
    criterion = nn.MSELoss()
    # 定义优化器：使用Adam优化器，初始学习率设置为1e-3（文档S12参数）
    optimizer = Adam(model.parameters(), lr=1e-3)
    # 定义学习率调度器：使用余弦退火调度，训练轮数T_max=100（文档S12要求）
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    # 初始化最佳验证损失为无穷大，用于保存性能最优的模型（避免过拟合）
    best_val_loss = float('inf')
    # 打印训练开始信息
    print(f"\n开始SSA模型训练（共100轮，设备={device}，SSA单位=m²/g）")

    # 训练循环，共执行100轮（文档S12要求）
    for epoch in range(100):
        # 训练阶段：将模型设置为训练模式
        model.train()
        # 初始化训练损失累计变量
        train_loss = 0.0
        # 遍历训练集的每个批次
        for batch_x, batch_y in train_loader:
            # 将批次数据移动到计算设备
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # 清零优化器的梯度
            optimizer.zero_grad()
            # 将输入传入模型，获取预测值（忽略特征输出）
            pred_y, _ = model(batch_x)
            # 计算预测值与真实标签的MSE损失
            loss = criterion(pred_y, batch_y)
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 累计批次损失（乘以批次大小，便于后续计算平均损失）
            train_loss += loss.item() * batch_x.size(0)
        # 计算平均训练损失（总损失除以训练集样本数）
        avg_train_loss = train_loss / len(train_loader.dataset)

        # 验证阶段：将模型设置为评估模式（关闭dropout等训练特有的层）
        model.eval()
        # 初始化验证损失累计变量
        val_loss = 0.0
        # 关闭梯度计算，节省内存并加速计算
        with torch.no_grad():
            # 遍历验证集的每个批次
            for batch_x, batch_y in val_loader:
                # 将批次数据移动到计算设备
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                # 获取模型预测值
                pred_y, _ = model(batch_x)
                # 计算损失
                loss = criterion(pred_y, batch_y)
                # 累计验证损失
                val_loss += loss.item() * batch_x.size(0)
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader.dataset)

        # 使用调度器更新学习率（余弦退火策略）
        scheduler.step()

        # 若当前验证损失小于最佳验证损失，更新最佳损失并保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_model_path)
            # 每20轮打印一次训练进度和模型保存信息
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d}/100 | 训练损失：{avg_train_loss:.6f} | 验证损失：{avg_val_loss:.6f} | 保存最优模型")

    # 训练完成后，加载保存的最优模型参数
    model.load_state_dict(torch.load(save_model_path))
    # 打印训练完成信息
    print(f"\nSSA模型训练完成：最优验证损失={best_val_loss:.6f} | 模型保存路径={save_model_path}")
    # 返回训练好的模型
    return model


# ---------------------- 4. 提取SSA的16维初步表征（文档S12核心产出） ----------------------
# 定义提取SSA的16维初步表征的函数
def extract_ssa_16d_repr(model, full_ssa_input, device, save_repr_path):
    # 将模型设置为评估模式，关闭梯度计算
    model.eval()
    # 将全量SSA输入数据移动到计算设备
    full_ssa_input = full_ssa_input.to(device)
    # 获取有效样本数量
    valid_samples = len(full_ssa_input)
    # 打印提取开始信息
    print(f"\n开始提取SSA的16维表征（单位：m²/g，样本数={valid_samples}）")

    # 分批提取表征（避免GPU内存不足），批次大小设为64
    batch_size = 64
    # 存储所有批次的16维表征
    all_16d_repr = []
    # 关闭梯度计算
    with torch.no_grad():
        # 按批次遍历全量数据
        for i in range(0, valid_samples, batch_size):
            # 获取当前批次的输入数据
            batch_x = full_ssa_input[i:i+batch_size]
            # 传入模型，仅获取特征提取器输出的16维特征（忽略预测值）
            _, batch_16d = model(batch_x)
            # 将16维特征从GPU转移到CPU，并转换为numpy数组，添加到列表中
            all_16d_repr.append(batch_16d.cpu().numpy())

    # 合并所有批次的表征，得到完整的16维表征数组
    full_16d_repr = np.concatenate(all_16d_repr, axis=0)
    # 对16维表征进行min-max归一化（文档S12要求），将特征标准化到[0,1]范围，适配后续PGDAE模型
    # 计算每个维度的最小值和最大值
    min_vals = full_16d_repr.min(axis=0)
    max_vals = full_16d_repr.max(axis=0)
    # 归一化公式：(x - min) / (max - min + 1e-8)，加1e-8避免分母为0
    full_16d_repr_norm = (full_16d_repr - min_vals) / (max_vals - min_vals + 1e-8)

    # 保存归一化后的16维表征（用于文档S15的PGDAE统一表征拼接）
    np.save(save_repr_path, full_16d_repr_norm)
    # 打印提取完成信息
    print(f"SSA 16维表征提取完成！\n- 表征形状：{full_16d_repr_norm.shape}（样本数×16维）\n- 保存路径：{save_repr_path}\n- 单位：m²/g（已归一化到[0,1]）")
    # 返回归一化后的16维表征
    return full_16d_repr_norm


# ---------------------- 5. 主函数（整合全流程，一键运行SSA表征构建） ----------------------
# 定义SSA表征构建的主函数
def main_ssa(excel_path, save_model_path, save_repr_path, ssa_col_name="SSA"):
    # 步骤1：设置计算设备（优先使用GPU，若无GPU则使用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 打印使用的计算设备
    print(f"使用计算设备：{device}")

    # 步骤2：执行SSA数据预处理（读取Excel文件并进行清洗）
    train_loader, val_loader, test_loader, full_ssa_input = preprocess_ssa_data(
        excel_path=excel_path,
        ssa_col_name=ssa_col_name,
        ssa_unit="m²/g"  # 固定为文档标准单位，无需修改
    )

    # 步骤3：训练SSA回归模型
    trained_model = train_ssa_model(train_loader, val_loader, device, save_model_path)

    # 步骤4：提取并保存SSA的16维初步表征
    ssa_16d_repr = extract_ssa_16d_repr(trained_model, full_ssa_input, device, save_repr_path)

    # 返回提取的16维表征
    return ssa_16d_repr


# ---------------------- 6. 运行入口（需修改为你的实际文件路径） ----------------------
# 当脚本作为主程序运行时，执行以下代码
if __name__ == "__main__":
    # 配置文件路径（请根据实际数据存储位置修改！）
    EXCEL_SSA_PATH = "mof_candidate_ssa.xlsx"          # SSA数据的Excel文件路径（单位：m²/g）
    SAVE_MODEL_PATH = "ssa_regression_best.pth"        # SSA模型权重的保存路径
    SAVE_REPR_PATH = "ssa_16d_preliminary_repr.npy"    # SSA 16维表征的保存路径
    SSA_COL_NAME = "SSA"                               # Excel中SSA列的标题（若不同需修改，如"比表面积(m²/g)"）

    # 一键运行SSA的16维表征构建全流程
    ssa_16d_repr = main_ssa(
        excel_path=EXCEL_SSA_PATH,
        save_model_path=SAVE_MODEL_PATH,
        save_repr_path=SAVE_REPR_PATH,
        ssa_col_name=SSA_COL_NAME
    )
