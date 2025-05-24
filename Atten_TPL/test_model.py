import os
import json
import torch
import model_Atten_TPL
import numpy as np
import re
from typing import Dict, List
from utility.parse_args import arg_parse


args = arg_parse()

app_functions = torch.load(args.function_feature_path)
def encode_test_data(data: Dict) -> Dict:
    tpl_list = data['tpl_list']
    if not tpl_list:
        tpl_list = [0]

    batch_function_feature = []
    batch_target_tpl = []
    batch_tpl_context = []
    batch_context_len = []

    tpl_context = tpl_list
    function_feature = app_functions[data['app_id']].to(args.device)
    context_len = len(tpl_list)

    for i in range(args.tpl_range):
        target_tpl = i + 1
        if target_tpl not in tpl_list:
            batch_target_tpl.append(target_tpl)
            batch_function_feature.append(function_feature)
            batch_tpl_context.append(tpl_context)
            batch_context_len.append(context_len)

    input_data = {
        'tpl_context': torch.tensor(batch_tpl_context).to(args.device),
        'function': torch.stack(batch_function_feature).to(args.device),
        'target_tpl': torch.tensor(batch_target_tpl).to(args.device),
        'context_len': torch.tensor(batch_context_len, dtype=torch.float32).to(args.device)
    }

    return input_data


def get_top_n_tpl(probability_list, top_n) -> List:
    p_list = probability_list
    p_list = sorted(p_list, reverse=True)
    top_n_tpl = []
    for i in range(top_n):
        top_n_tpl.append(p_list[i][1])
    return top_n_tpl


def test_model(model_path: str) -> None:
    recommend_file = args.output_path + output_file

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    write_recommend_fp = open(file=recommend_file, mode='w')
    test_fp = open(file=args.testing_data_path + test_dataset, mode='r')

    model = model_Atten_TPL.AttenTPL()
    model.load_state_dict(torch.load(model_path))

    model = model.to(args.device)
    result_list: List = []

    test_num: int = 0
    model.eval()
    with torch.no_grad():
        for lines in test_fp.readlines():
            test_obj = json.loads(lines.strip('\n'))
            # print(test_obj)
            test_num += 1
            inputs = encode_test_data(test_obj)

            outputs = model(inputs)
            outputs = outputs.view(-1).tolist()
            probability_list = []

            tpl_list = test_obj['tpl_list']

            num = 0
            for i in range(args.tpl_range):
                target_tpl = i + 1
                if target_tpl not in tpl_list:
                    probability_list.append((outputs[num], target_tpl))
                    num += 1

            top_n_tpl = get_top_n_tpl(probability_list, 20)
            write_data = {
                'app_id': test_obj['app_id'],
                'removed_tpls': test_obj['removed_tpl_list'],
                'recommend_tpls': top_n_tpl,
                'tpl_list': test_obj['tpl_list'],

            }
            write_content = json.dumps(write_data) + '\n'
            write_recommend_fp.write(write_content)

    test_fp.close()
    write_recommend_fp.close()


if __name__ == '__main__':
    model_path = 'model_Atten_TPL'
    test_dataset = args.test_dataset
    pattern = r"testing_(\d+)_(\d+)\.json"
    matches = re.match(pattern, test_dataset)
    if matches:
        fold = matches.group(1)
        rm = matches.group(2)
    output_file = 'testing_Atten_TPL_'+str(fold)+'_'+str(rm)+ '.json'
    test_model('./' + model_path + '/model_' + str(fold) + '.pth')

