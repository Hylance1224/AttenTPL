import os
import re
import json
import utility.config
import numpy as np

import utility.utils as utils
from typing import Dict, Tuple

args = utility.config.args


def load_relation_mat(train_file_path: str) -> np.ndarray:
    fold_rmv = re.findall(r'\d+', args.training_dataset)
    relation_file_name = 'relation_%s_%s.txt' % (fold_rmv[0], fold_rmv[1])

    if os.path.exists(args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + relation_file_name):
        relation = np.loadtxt(args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + relation_file_name, dtype=np.int8)
        return relation

    size_apk: int = 0
    size_lib: int = 0

    train_fp = open(file=train_file_path, mode='r')
    for line in train_fp.readlines():
        obj = json.loads(line.strip('\n'))

        app_id = obj['app_id']
        tpl_list = obj['tpl_list']
        if len(tpl_list) == 0:              # 加这个的目的是如果tpl为空的情况，就不放进去训练了，直接在测试的时候给他一个800就行
            continue

        ###########################################
        size_apk = max(size_apk, app_id)
        size_lib = max(max(tpl_list), size_lib)
        ###########################################
    size_apk += 1
    size_lib += 1
    relation = np.zeros(shape=(size_apk, size_lib), dtype=np.int8)

    train_fp.seek(0)
    for line in train_fp.readlines():
        obj = json.loads(line.strip('\n'))
        app_id = obj['app_id']
        tpl_list = obj['tpl_list']

        for tpl in tpl_list:
            relation[app_id, tpl_list] = 1

    utils.ensure_dir(args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/')
    np.savetxt(fname=args.relation_path + f'fold{fold_rmv[0]}_rmv{fold_rmv[1]}/' + relation_file_name, X=relation, fmt='%d')

    return relation
