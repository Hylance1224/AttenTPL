import random
import json
import os
from utility.parse_args import arg_parse

args = arg_parse()

fold_num = 5
removed_nums = [1, 3, 5]


def read_json_relation_file(path='data/relation_shuffle.json'):
    f = open(path, 'r')
    lines = f.readlines()
    datas = []
    for line in lines:
        data = json.loads(line.strip())
        datas.append(data)
    f.close()
    return datas


def generate_training_data(AppID_TplList, file='data/train.json'):
    f = open(file, 'w', encoding='utf-8')
    for data in AppID_TplList:
        app_id = data['app_id']
        tpl_list = data['tpl_list']
        for i in range(0, len(tpl_list)):
            # generate true sample
            new_tpl_list = tpl_list.copy()
            new_tpl_list.pop(i)
            temp = {'app_id': app_id, 'tpl_list': new_tpl_list, 'target_tpl': tpl_list[i], 'label': 1}

            js = json.dumps(temp, ensure_ascii=False)
            f.write(js)
            f.write('\n')

            # generate false sample
            false_tpl_list = [x for x in range(1, args.tpl_range + 1) if x not in tpl_list]
            false_tpls = random.choices(false_tpl_list, k=10)
            for i in false_tpls:
                false_tpl = i
                temp = {'app_id': app_id, 'tpl_list': new_tpl_list, 'target_tpl': false_tpl,
                         'label': 0}
                js = json.dumps(temp, ensure_ascii=False)
                f.write(js)
                f.write('\n')
    f.close()


def generate_testing_data(AppID_TplList, remove_num, file='data/test.json'):
    with open(file, 'w', encoding='utf-8') as f:
        for data in AppID_TplList:
            app_id = data['app_id']
            tpl_list = data['tpl_list']
            if len(tpl_list) >= remove_num:
                tpl_list_copy = tpl_list.copy()
                removed_tpl_list = random.sample(tpl_list_copy, remove_num)
                for r in removed_tpl_list:
                    tpl_list_copy.remove(r)
                temp = {'app_id': app_id, 'tpl_list': tpl_list_copy, 'removed_tpl_list': removed_tpl_list}
                js = json.dumps(temp, ensure_ascii=False)
                f.write(js + '\n')



def main_training(json_relation_file='data/relation_shuffle.json', n_fold=fold_num):
    AppID_TplList = read_json_relation_file(json_relation_file)
    length = len(AppID_TplList)
    fold_num = int(length/n_fold)
    for i in range(n_fold):
        datas = AppID_TplList[0:fold_num*i]
        datas.extend(AppID_TplList[fold_num*(i+1):])
        generate_training_data(datas, args.training_data_path + '/training_' + str(i) + '.json')


def main_testing(json_relation_file='data/relation_shuffle.json', n_fold=fold_num):
    AppID_TplList = read_json_relation_file(json_relation_file)
    length = len(AppID_TplList)
    fold_num = int(length/n_fold)
    for i in range(n_fold):
        for j in removed_nums:
            temp = AppID_TplList[fold_num * i: fold_num * (i + 1)]
            generate_testing_data(temp, remove_num=j,
                                  file=args.testing_data_path + '/testing_' + str(i) + '_' + str(j) + '.json')


if __name__ == '__main__':
    if not os.path.exists(args.training_data_path):
        os.mkdir(args.training_data_path)
    if not os.path.exists(args.testing_data_path):
        os.mkdir(args.testing_data_path)
    main_training()
    main_testing()


