"""
EB(Escalator bench) 数据集
NC IRF ORF REF CF 5 大类
工频0.15Hz, fs=12.8kHz, each file contains 1280000 points.
128 0000 // 2048 = 625
"""
import os

_dir = r'E:\科研\datas\EB\EB_data'


def get_filename(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    return file_list


# normal condition
NC = get_filename(os.path.join(_dir, 'NC'))

# inner race fault
IRF1 = get_filename(os.path.join(_dir, r'IRF\IRF1'))
IRF2 = get_filename(os.path.join(_dir, r'IRF\IRF2'))
IRF3 = get_filename(os.path.join(_dir, r'IRF\IRF3'))

# outer race fault
ORF1 = get_filename(os.path.join(_dir, r'ORF\ORF1'))
ORF2 = get_filename(os.path.join(_dir, r'ORF\ORF2'))
ORF3 = get_filename(os.path.join(_dir, r'ORF\ORF3'))

# rolling element fault
REF1 = get_filename(os.path.join(_dir, r'REF\REF1'))
REF2 = get_filename(os.path.join(_dir, r'REF\REF2'))
REF3 = get_filename(os.path.join(_dir, r'REF\REF3'))

# cage fault
CF1 = get_filename(os.path.join(_dir, r'CF\CF1'))
CF2 = get_filename(os.path.join(_dir, r'CF\CF2'))
CF3 = get_filename(os.path.join(_dir, r'CF\CF3'))

# Cases
EB_3way_1 = [NC[0], IRF1[0], ORF1[0]]
EB_3way_2 = [NC[0], IRF2[0], ORF2[0]]
EB_3way_3 = [NC[0], IRF3[0], ORF3[0]]

EB_4way_1 = [NC[0], IRF1[0], ORF1[0], REF1[0]]
EB_4way_2 = [NC[0], IRF2[0], ORF2[0], REF2[0]]
EB_4way_3 = [NC[0], IRF3[0], ORF3[0], REF3[0]]

EB_5way_1 = [NC[0], IRF1[0], ORF1[0], REF1[0], CF1[0]]
EB_5way_2 = [NC[0], IRF2[0], ORF2[0], REF2[0], CF2[0]]

EB_13way = [NC[0], IRF1[0], IRF2[0], IRF3[0], ORF1[0], ORF2[0], ORF3[0],
            REF1[0], REF2[0], REF3[0], CF1[0], CF2[0], CF3[0]]

EB_8way_1 = [ IRF2[0], IRF3[0], ORF2[0], ORF3[0], REF2[0], REF3[0], CF2[0], CF3[0]]

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
    print(EB_3way_3)
