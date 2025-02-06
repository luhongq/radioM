import math

import numpy as np
from sklearn.model_selection import train_test_split
from  tqdm import  tqdm
import warnings
from scipy.optimize import minimize
# 屏蔽 FutureWarning 警告
warnings.filterwarnings("ignore", category=FutureWarning)
colun=['FB','RSP', 'CCI', 'Hb', 'Husr', 'Hm',
             'deltaH', 'L', 'D',
           'UCI', 'SINR','dg','dr','dw','hv','db','do','LOS']
colunyuanshi=['FB','RSP', 'CCI', 'Hb', 'Husr', 'Hm',
             'deltaH', 'L', 'D',
           'UCI', 'SINR','dg','dr','dw','hv','db','do','LOS','RSRP']
import os
import pandas as pd
import time
import ast

#经验模型验证

def get_filename(root_dir):
    filenames = []

    # 遍历所有子目录和文件
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            if name.endswith('.csv'):  # 只选择CSV文件

                file_name = name
                file_content = os.path.join(root, name)
                filenames.append([file_name, file_content])
    return filenames


#读取pickle数据集
def load_dataset(dataset_path, debug=False):
    t = time.time()
    sample_cnt = 0


    # 获取所有文件的路径
    path = get_filename(dataset_path)
    all_data = pd.DataFrame(columns=colun)
    # 遍历每个文件
    for file_name, file_content in tqdm(path):
        sample_cnt += 1


        with open(file_content, 'rb') as test:
            pb_data = pd.read_pickle(test)


        # 只选择需要的列
        pb_data = pb_data[colun]


        # 将该文件的数据追加到list中
        all_data = pd.concat([all_data, pb_data], ignore_index=True)

        if debug:
            if sample_cnt == 1:  # 如果debug模式开启，只处理一个文件
                break


    print((all_data['deltaH'] < 0).sum())

    return all_data



#整体模型计算类

class FormulaModel:

    def __init__(self, formula, selected_features,model_name):
        """
        初始化模型，设定数学公式和所选特征
        :param formula: 一个函数，定义了预测的数学公式
        :param selected_features: 需要用于公式计算的特征名称列表
        """
        self.formula = formula
        self.selected_features = selected_features
        print(f'现在使用{model_name}模型预测')

    def fit(self, X_train, y_train):
        """
        模拟模型的训练过程
        :param X_train: 训练数据（DataFrame）
        :param y_train: 训练标签
        """
        start_time = time.time()
        # 只选择指定的特征用于训练
        X_train_selected = X_train[self.selected_features]
        # 构建设计矩阵（包含偏置项1的列，便于拟合常数项）
        A = np.column_stack([np.ones(X_train_selected.shape[0]), X_train_selected])

        # 最小二乘法求解
        self.coefficients_, residuals, rank, s = np.linalg.lstsq(A, y_train, rcond=None)
        print(f"拟合的系数: {self.coefficients_}")
        print(f"训练中，使用公式 {self.formula.__name__} 并选择特征 {self.selected_features}")
        end_time = time.time()
        print(f"训练完成，耗时 {end_time - start_time:.2f} 秒")

    def predict(self, X):
        """
        使用公式进行预测
        :param X: 输入特征数据（DataFrame）
        :return: 预测值
        """
        # 只选择指定的特征用于预测

        X_selected = X[self.selected_features]
        return self.formula(X_selected)

    def score(self, X_test, y_test):
        """
        测试模型的效果
        :param X_test: 测试数据特征（DataFrame）
        :param y_test: 测试数据标签
        :return: 模型的平均绝对误差
        """
        start_time = time.time()
        X_test=X_test[self.selected_features]
        y_pred = self.predict(X_test).reshape(-1, 1)
        y_test=np.array(y_test.values)
        # print(y_pred,y_test)
        error = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        mape = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

        end_time = time.time()
        print(f"测试完成，耗时 {end_time - start_time:.2f} 秒")
        print(f"平均绝对误差: {error:.4f}")
        print(f"均方根误差: {rmse:.4f}")
        print(f"平均绝对百分比误差: {mape:.4f}")
        return error


#Cost231模型
def cost231(X):

    # 计算 ah
    # ah = (1.1 * np.log10(X['FB'].values) - 0.7) * (1.5 + X['Hm'].values) - (1.56 * np.log10(X['FB'].values) - 0.8)
    ah=3.2*(np.log10(11.75*(X['Hm'].values+1.5)))**2-4.97
    # 初始化 cell 和 cm 列
    cell = np.zeros(X.shape[0])
    cm = np.zeros(X.shape[0])

    # 对 UCI 列应用条件判断
    uci_conditions = X['UCI'].isin([4, 19])
    cell[uci_conditions] = -4.78 * (np.log10(X['FB'].values[uci_conditions]) ** 2) - 18.33 * np.log10(X['FB'].values[uci_conditions]) - 40.98
    cm[uci_conditions] = 0

    uci_conditions = X['UCI'].isin([5, 10, 11, 12, 13, 14, 15, 16, 20])
    cell[uci_conditions] = 0
    cm[uci_conditions] = 3

    uci_conditions = ~X['UCI'].isin([4, 19, 5, 10, 11, 12, 13, 14, 15, 16, 20])
    cell[uci_conditions] = -2 * (np.log10(X['FB'].values[uci_conditions] / 28) )** 2 - 5.4
    cm[uci_conditions] = 0

    # 计算 Lp
    Lp = 46.3 + 33.9 * np.log10(X['FB'].values) +(44.96 - 6.55 * np.log10(X['Hb'].values)) * np.log10(X['D'].values / 1000) -13.82 * np.log10(X['Hb'].values) - ah + cm + cell
    # print(Lp)
    # 返回 RSP 和 Lp 的差值
    return np.array((X['RSP'].values - Lp).ravel())
#SPM模型
def spm(X):
    # Ldif=0
    # Lp =K1+K2*np.log10(X['D'].values)+K3*np.log10(X['deltaH'].values-1.5-X['Hm'].values)+K4*Ldif+K5*np.log10(X['deltaH'].values)*np.log10(X['D'].values)+K6*(1.5 + X['Hm'].values)+K7*
    clutter = np.zeros(X.shape[0])

    clutter_condition=X['UCI'].isin([ 10, 11, 12, 13, 14, 15, 16])
    clutter[clutter_condition]=3.37


    Lp=23.5+44.9*np.log10(X['D'].values)-6.55*np.log10(X['Hb'].values)*np.log10(X['D'].values)+5.83*np.log10(np.abs(X['Hb'].values-(X['Husr'].values+ X['Hm'].values)-1.5))
    # print(Lp)
    return np.array((X['RSP'].values - Lp).ravel())


    Gt=0
    Gr=0
    f,hteff,hr,data1=X['FB'].values,X['deltaH'].values,X['Hm'].values+1.5,X['ekparam'].values

    a0=np.where(X['UCI'].isin([4, 17, 18, 19]), 1.5,
               np.where(X['UCI'].isin([1, 2, 3, 7, 8, 9]), 0.5, 2.5))
    X['y'] = X['y'].astype(float)
    X['x'] = X['x'].astype(float)
    h_string =X['deltaH'].values-X['Hm'].values-1.5
    a = np.where((X['x'] == 0) & (X['y'] == 0), 0,
                 np.where(X['x'] == 0, 90,
                 np.degrees(np.arctan(np.divide(X['y'].values, X['x'].values,
                               out=np.zeros_like(X['y'].values), where=X['x'].values != 0)))))
    d = X['D'].values/ 1000
    dmod = [min(max(5, value), 100) for value in d]
    amod =  [min(50, value) for value in a]

    r1 = 4.49 - 0.655 * np.log10(hteff)
    lamda = 300 / f
    Ad = []
    Adif = []
    e1 = []
    for data,r2,d1,lamda1,hr1 in tqdm(zip(data1,r1,d,lamda,hr)):  # 每行数据是一个包含多个字典的列表
        data=ast.literal_eval(data)
        r = [r2]
        dis = []
        h_bui = []


        for i in range(0, len(data), 2):
            dis.append((data[i]['distance'] + data[i + 1]['distance']) / 2000)
            h_bui.append((data[i]['elevation'] + data[i + 1]['elevation']) / 2)

        dis.append(d1)

        for i in range(len(dis) - 1):
            d_mid = (dis[i] + dis[i + 1]) / 2

            ci = -(h_string - h_bui[i])
            vi = ci * np.sqrt(2 * d1 / (lamda1 * d_mid * (d1 - d_mid)))
            r.append(2 + (r2 - 2) * (1 + np.tanh(2 * (vi + 1)) / 2))

        if len(data) / 2 == 1:
            Adif.append(5)
        elif len(data) / 2 == 2:
            Adif.append(9)
        elif len(data) / 2 == 3:
            Adif.append(12)
        elif len(data) / 2 == 4:
            Adif.append(14)
        elif len(data) / 2 >= 5:
            Adif.append(15)
        else:
            Adif.append(0)

        if hr1 <= 5:
            e1.append(10)
        elif 5 < hr1 < 10:
            e1.append(2 * hr1)
        else:
            e1.append(20)
        Ad.append(calculateExpression(r, dis)-(5*np.log10(5*d1+ 1) + 2))



    At = -13.82 * np.log10(hteff)

    Ar = -3 - e1 * np.log10(hr * e1 / 3)

    L0 = 69.6 + 26.2 * np.log10(f) + Ad + Ar + At + Gt + Gr
    print(L0, Ad, Ar, At)

    Alu = 1



    Aor = a0 * (amod - 35) * (1 + np.log10(10 / dmod)) / 25
    # print(L0,Adif,Aor)
    return np.array(X['RSP'].values-(L0 + Adif + Alu + Aor))






def rmse(params, X, y_true):
    # 打印当前参数
    print(f"当前参数: {params}")
    y_pred = ABG(X, params)

    return np.sqrt(np.mean((np.array(y_true.values).reshape(-1, 1) - np.array(y_pred).reshape(-1, 1)) ** 2))


#  TR38.901模型
def tr38901(X):
    X.loc[:, 'deltaH'] = np.where(X['deltaH'] == 0, X['deltaH'] + 1, np.abs(X['deltaH']))

    dBP=4*(X['deltaH'].values)*X['Hm'].values*X['FB'].values/300


    ta1= np.where(
                    (X['L'].values > 10) & (X['L'].values < dBP),
                    28 + 22 * np.log10(X['D'].values) + 20 * np.log10(X['FB'].values / 1000),
                    28 + 40 * np.log10(X['D'].values) + 20 * np.log10(X['FB'].values / 1000) -
                    9 * np.log10(dBP ** 2 + (X['deltaH'].values- X['Hm'].values) ** 2)
                    )
    ta2=13.54+39.08*np.log10(X['D'].values)+20*np.log10(np.abs((X['FB'].values / 1000)-0.6*(X['Hm'].values))+0.1)
    ti1= np.where(
                X['L'].values <= 10,
                32.4 + 21 * np.log10(X['D'].values) + 20 * np.log10(X['FB'].values / 1000),
                np.where(
                    (X['L'].values > 10) & (X['L'].values < dBP),
                    32.4 + 21 * np.log10(X['D'].values) + 20 * np.log10(X['FB'].values / 1000),
                    32.4 + 40 * np.log10(X['D'].values) + 20 * np.log10(X['FB'].values / 1000) -
                    9.5 * np.log10(dBP ** 2 + (X['deltaH'].values - X['Hm'].values ) ** 2)
                )
            )
    ti2=22.4+35.3*np.log10(X['D'].values)+21.3*np.log10(np.abs((X['FB'].values / 1000)-0.3*(X['Hm'].values)))
    # ti2=32.4+31.9*np.log10(X['D'].values)+20*np.log10(X['FB'].values / 1000)+ + np.random.normal(0, 8.2)
    pl = np.where(X['deltaH'].values >= 0,
                  np.where(X['LOS'].values,ta1,np.maximum(ta1,ta2)),
                  np.where(X['LOS'].values,ti1,np.maximum(ti1,ti2))
                  )




    # print(loss)
    return np.array(X['RSP'].values-pl)


##填入CSV文件经验模型的预测结果
def process_csv_files_in_directory(directory_path,out_path):
    #获取目录下的所有文件列表
    files = os.listdir(directory_path)
    # 累积 RMSE 和 MAPE 计算变量
    # 累积每种模型的误差数据
    model_error_stats = {
        'COST231': {'squared_error': 0, 'absolute_percentage_error': 0, 'sample_count': 0},
        'SPM': {'squared_error': 0, 'absolute_percentage_error': 0, 'sample_count': 0},
        'TR38901': {'squared_error': 0, 'absolute_percentage_error': 0, 'sample_count': 0},
    }
    # 遍历所有 CSV 文件
    for file in tqdm(files):
        if file.endswith('.csv'):

            file_path = os.path.join(directory_path, file)

            # 读取 CSV 文件
            data = pd.read_csv(file_path)

            
        
            # 数据类型转换
            colunfloat = ['FB', 'RSP', 'L', 'D', 'RSRP']
            colunint = ['CCI', 'Hb', 'Husr', 'Hm', 'deltaH', 'UCI']
            data[colunfloat] = data[colunfloat].astype("float")
            data[colunint] = data[colunint].astype('int')

            # 遍历每种模型进行预测和误差累积
            for model_name, model_func, selected_features in [
                ('COST231', cost231, ['FB', 'deltaH', 'UCI', 'D', 'RSP', 'Hm', 'Husr', 'Hb']),
                ('SPM', spm, ['deltaH', 'FB', 'D', 'RSP', 'Hm', 'UCI', 'Husr', 'Hb']),
                ('TR38901', tr38901, ['deltaH', 'FB', 'D', 'L', 'RSP', 'Hm', 'LOS', 'Husr', 'Hb']),
            ]:
                # 创建模型实例并预测
                model = FormulaModel(model_func, selected_features=selected_features, model_name=model_name)
                data[model_name] = model.predict(data)

                # 计算误差
                squared_error = (data[model_name] - data['RSRP']) ** 2
                absolute_percentage_error = np.abs((data[model_name] - data['RSRP']) / data['RSRP']) * 100

                # 累积到对应模型的误差统计
                model_error_stats[model_name]['squared_error'] += squared_error.sum()
                model_error_stats[model_name]['absolute_percentage_error'] += absolute_percentage_error.sum()
                model_error_stats[model_name]['sample_count'] += len(data)

            # 保存处理后的结果到新文件
            result_file_path = os.path.join(out_path, file)
            data.to_csv(result_file_path, index=False)

    # 最终计算每种模型的总 RMSE 和总 MAPE
    print("\n========== 各模型总误差统计 ==========")
    for model_name, stats in model_error_stats.items():
        total_rmse = np.sqrt(stats['squared_error'] / stats['sample_count'])
        total_mape = stats['absolute_percentage_error'] / stats['sample_count']
        print(f"{model_name}: 总 RMSE = {total_rmse:.4f}, 总 MAPE = {total_mape:.2f}%")









#填入经验模型预测结果
dataset_path = r'dataset/radioM/'
out_path='result/'  
process_csv_files_in_directory(dataset_path,out_path)
#填入机器学习模型预测结果
# process_csv_files_in_directory1(dataset_path,out_path)



##测试经验性模型
# if __name__ == '__main__':
#
#     dataset_path = 'F:/wireless/huawei/filter_dataset/'
#     data = load_dataset(dataset_path, debug=False)
#     print(dataset_path)
#     data = data.dropna(axis=0, how='any')
#
#     print("len(data):", len(data))
#
#
#
#
#
#     colunfloat=['FB', 'RSP','betaV', 'deltaHv', 'L', 'D','RSRP']
#     colunint=['CCI', 'Hb', 'Husr', 'Hm','deltaH','UCI','x','y']
#     colunlist=['ekparam']
#     data[colunfloat]=data[colunfloat].astype("float")
#     data[colunint]=data[colunint].astype('int')
#
#
#
#     inputs = data[colun]
#
#     label = data[['RSRP']]





    # model_name='cost231'
    # # 创建模型实例，传入数学公式和选定的特征
    # model = FormulaModel(cost231, selected_features=['FB','deltaH','UCI','D','RSP','Hm','Husr','Hb'],model_name=model_name)
    # model.score(inputs, label)
    #
    # model_name='spm'
    # model = FormulaModel(spm, selected_features=['deltaH', 'FB', 'D', 'RSP', 'Hm','UCI','Husr','Hb'],model_name=model_name)
    # model.score(inputs, label)
    #
    # model_name = 'tr38901'
    # model = FormulaModel(tr38901, selected_features=['deltaH', 'FB', 'D', 'L', 'RSP', 'Hm', 'LOS','Husr','Hb'],
    #                      model_name=model_name)
    # model.score(inputs, label)


    # X_train, X_test, Y_train, Y_test = \
    #     train_test_split(inputs, label, test_size=0.2, shuffle=True)
    #
    # # 打印输入数据的形状和第一个数据行
    # print("训练集形状:", X_train.shape)
    #
    # # 打印标签数据的形状和第一个数据行
    # print("测试集形状:",X_test.shape )
    # model_name='ABG'
    # x_initial = np.array([4, 10.2, 2.36, 7.6])
    # bounds = [(2, 4), (0, 20), (2, 3), (3, 12)]
    # args = (X_train, Y_train)
    # result = minimize(rmse, x_initial, args=args, method='Powell', bounds=bounds)
    #
    # a_fit, b_fit, r_fit, o_fit = result.x
    # print(f'拟合参数: a = {a_fit}, b = {b_fit}, r = {r_fit}, o = {o_fit}')
    # y_fit = simulate_model_rmse([a_fit, b_fit, r_fit, o_fit],X_test,Y_test, 1000)