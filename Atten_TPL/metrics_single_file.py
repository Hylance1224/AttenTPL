import json
from utility.parse_args import arg_parse

args = arg_parse()


def calculate_Precision(recommend_list, removed_tpl_list, top_n):
    hit = 0.0
    if len(recommend_list) == 0:
        return hit
    for i in range(top_n):
        if recommend_list[i] in removed_tpl_list:
            hit = hit + 1
    return hit/top_n


def calculate_Recall(recommend_list, removed_tpl_list, top_n):
    hit = 0.0
    if len(recommend_list) == 0:
        return hit
    for i in range(top_n):
        if recommend_list[i] in removed_tpl_list:
            hit = hit + 1
    return hit/len(removed_tpl_list)


def calculate_AP(recommend_list, removed_tpl_list, top_n):
    cor_list = []
    if len(recommend_list) == 0:
        return 0
    for i in range(top_n):
        if recommend_list[i] in removed_tpl_list:
            cor_list.append(1.0)
        else:
            cor_list.append(0.0)
    sum_cor_list = sum(cor_list)
    if sum_cor_list == 0:
        return 0

    summary = 0
    for i in range(top_n):
        t = (sum(cor_list[:i+1])/(i+1))*(cor_list[i])
        summary = summary + t
    return summary/sum_cor_list



def read_json_file(path):
    f = open(path, 'r')
    lines = f.readlines()
    datas = []
    for line in lines:
        data = json.loads(line.strip())
        datas.append(data)
    f.close()
    return datas


def calculate_metrics(datas, tpl_num):
    all_precision_5 =[]
    all_precision_10 = []
    all_recall_5 = []
    all_recall_10 = []
    all_AP_5 = []
    all_AP_10 = []
    cov_5_tpls = []
    cov_10_tpls = []
    num = 1
    for data in datas:
        recommend_list = data['recommend_tpls']
        removed_tpl_list = data['removed_tpls']
        precision_5 = calculate_Precision(recommend_list, removed_tpl_list, RecNum_1)
        all_precision_5.append(precision_5)
        num=num+1

        precision_10 = calculate_Precision(recommend_list, removed_tpl_list, RecNum_2)
        all_precision_10.append(precision_10)
        recall_5 = calculate_Recall(recommend_list, removed_tpl_list, RecNum_1)
        all_recall_5.append(recall_5)
        recall_10 = calculate_Recall(recommend_list, removed_tpl_list, RecNum_2)
        all_recall_10.append(recall_10)
        AP_5 = calculate_AP(recommend_list, removed_tpl_list, RecNum_1)
        all_AP_5.append(AP_5)
        AP_10 = calculate_AP(recommend_list, removed_tpl_list, RecNum_2)
        all_AP_10.append(AP_10)
        for tpl in recommend_list[:RecNum_1]:
            if tpl not in cov_5_tpls:
                cov_5_tpls.append(tpl)
        for tpl in recommend_list[:RecNum_2]:
            if tpl not in cov_10_tpls:
                cov_10_tpls.append(tpl)
    COV_5 = len(cov_5_tpls) / tpl_num
    COV_10 = len(cov_10_tpls) / tpl_num
    Mean_Precision_5 = sum(all_precision_5)/len(all_precision_5)
    Mean_Recall_5 = sum(all_recall_5) / len(all_recall_5)
    if Mean_Precision_5 == 0:
        Mean_F1_5 = 0
    else:
        Mean_F1_5 = (2*Mean_Precision_5*Mean_Recall_5)/(Mean_Precision_5+Mean_Recall_5)
    MAP_5 = sum(all_AP_5) / len(all_AP_5)
    Mean_Precision_10 = sum(all_precision_10) / len(all_precision_10)
    Mean_Recall_10 = sum(all_recall_10) / len(all_recall_10)
    if Mean_Precision_10 == 0:
        Mean_F1_10 = 0
    else:
        Mean_F1_10 = (2 * Mean_Precision_10 * Mean_Recall_10) / (Mean_Precision_10 + Mean_Recall_10)
    MAP_10 = sum(all_AP_10) / len(all_AP_10)
    # print('%2f %2f %2f %2f %2f %2f %2f %2f %2f %2f' % (Mean_Precision_5, Mean_Recall_5, Mean_F1_5, MAP_5, COV_5,
    #                                                    Mean_Precision_10, Mean_Recall_10, Mean_F1_10, MAP_10, COV_10))

    print('Top-rm  => MP: %2f  MR: %2f  MF: %2f  MAP: %2f  COV: %2f' % (
        Mean_Precision_5, Mean_Recall_5, Mean_F1_5, MAP_5, COV_5))
    print('Top-2rm => MP: %2f  MR: %2f  MF: %2f  MAP: %2f  COV: %2f' % (
        Mean_Precision_10, Mean_Recall_10, Mean_F1_10, MAP_10, COV_10))


def calculate_tpl_num(train_datas):
    tpl_list = []
    for data in train_datas:
        tpl = data['target_tpl']
        if tpl not in tpl_list:
            tpl_list.append(tpl)
    return len(tpl_list)


if __name__ == '__main__':
    tpl_num = args.tpl_range
    datas = []
    method = 'Atten_TPL'
    fold = args.fold
    rm_num = args.rm
    RecNum_1 = rm_num
    RecNum_2 = 2*rm_num
    test = 'testing_'
    output_path = args.output_path
    data = read_json_file(f"{output_path}/{test}{method}_{fold}_{rm_num}.json" )
    calculate_metrics(data, tpl_num)



