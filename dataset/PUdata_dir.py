"""
PU数据集
NC IR OR IROR 4大类
"""
import os

_dir = r'E:\科研\datas\PU\PUdata'


def get_filename(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    # if len(file_list) != 1:  # 若文件中存在不止一个文件，则存在歧义
    #     print('There are {} files in [{}]'.format(len(file_list), root_path))
    #     exit()
    return file_list


# normal condition
NC = get_filename(os.path.join(_dir, r'NC\K004'))

# inner race fault
IR1 = get_filename(os.path.join(_dir, r'IR\KI03'))
IR2 = get_filename(os.path.join(_dir, r'IR\KI07'))
IR3 = get_filename(os.path.join(_dir, r'IR\KI05'))
IR4 = get_filename(os.path.join(_dir, r'IR\KI08'))

# outer race fault
OR1 = get_filename(os.path.join(_dir, r'OR\KA05'))
OR2 = get_filename(os.path.join(_dir, r'OR\KA03'))
OR3 = get_filename(os.path.join(_dir, r'OR\KA01'))
OR4 = get_filename(os.path.join(_dir, r'OR\KA07'))
OR5 = get_filename(os.path.join(_dir, r'OR\KA07'))
OR6 = get_filename(os.path.join(_dir, r'OR\KA08'))

# inner race fault and outer race fault
IROR1 = get_filename(os.path.join(_dir, r'IROR\KB23'))
IROR2 = get_filename(os.path.join(_dir, r'IROR\KB27'))

# Cases
# EB_3way_1 = [NC[0], IR1[0], OR1[0]]
# EB_3way_2 = [NC[0], IR2[0], OR2[0]]
# EB_3way_3 = [NC[0], IR3[0], OR3[0]]
#
# EB_4way_1 = [NC[0], IR1[0], OR1[0], REF1[0]]
# EB_4way_2 = [NC[0], IR2[0], OR2[0], REF2[0]]
# EB_4way_3 = [NC[0], IR3[0], OR3[0], REF3[0]]
#
# EB_5way_1 = [NC[0], IR1[0], OR1[0], REF1[0], CF1[0]]
# EB_5way_2 = [NC[0], IR2[0], OR2[0], REF2[0], CF2[0]]
#
# EB_13way = [NC[0], IR1[0], IR2[0], IR3[0], OR1[0], OR2[0], OR3[0],
#             REF1[0], REF2[0], REF3[0], CF1[0], CF2[0], CF3[0]]



PU_3way_1 = [NC, IR1, OR1]
PU_3way_2 = [NC, IR2, OR2]
PU_3way_3 = [NC, IR3, OR3]
PU_3way_4 = [NC, IR4, OR4]



PU_5way_1 =[NC,IR1,IR2,IR3,IR4]
PU_8way_1 =[OR1,OR2,OR3,OR4,OR5,OR6,IROR1,IROR2]

PU_5way_2 =[NC,IR1,IROR1,OR1,IROR2]
PU_8way_2 =[OR1,OR2,OR3,OR4,OR5,OR6,IROR1,IROR2]

if __name__ == "__main__":
    # print(NC)
    # print(IR1)
    # print(IR2)
    # print(IR3)
    # print(OR1)
    # print(OR2)
    # print(OR3)
    # print(REF1)
    # print(REF2)
    # print(REF3)
    # print(CF1)
    # print(CF2)
    # print(CF3)
    print(PU_5way_1)
    print(PU_8way_1)
    print(PU_5way_2)
    print(PU_8way_2)
