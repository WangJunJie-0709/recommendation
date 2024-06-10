import csv
import os

def text_save(srcPath, dstPath, rows=None, column_name=None):
    '''
    将txt文件写入csv
    :param srcPath:
    :param dstPath:
    :return:
    '''
    if os.path.exists(dstPath):
        os.remove(dstPath)
    data = []
    with open(srcPath, 'r') as f:
        for i, line in enumerate(f):
            if rows is not None and i > rows:
                break
            data.append(list(line.strip().split('\t')))
    # 创建文件对象
    f = open(dstPath, 'w', encoding='utf-8', newline='')

    csv_writer = csv.writer(f)
    if column_name:
        csv_writer.writerow(column_name)
    csv_writer.writerows(data)

    f.close()
    return data


if __name__ == '__main__':
    # 写入物品内容
    item_src_path = r'../dataset/aliyun_tianchi/tianchi_fresh_comp_train_item_online/tianchi_fresh_comp_train_item_online.txt'
    item_col_name = ['item_id', 'item_geohash', 'item_category']
    item_dst_path = r'../tianchi_fresh_comp_train_item_online.csv'
    text_save(item_src_path, item_dst_path, rows=5000, column_name=item_col_name)

    # 写入用户与物品交互内容
    user_src_path = r'../dataset/aliyun_tianchi/tianchi_fresh_comp_train_user_online_partA/tianchi_fresh_comp_train_user_online_partA.txt'
    user_col_name = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']
    user_dst_path = r'../tianchi_fresh_comp_train_user_online.csv'
    text_save(user_src_path, user_dst_path, rows=1e3, column_name=user_col_name)