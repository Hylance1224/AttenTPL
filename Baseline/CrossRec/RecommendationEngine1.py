import os
import numpy as np
from utility.DataReader import *
from utility.parser import *
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class Recommendation:
    def __init__(self):
        self.parser = arg_parse()
        self.num_cols = 0
        self.num_rows = 0
        self.num_neighbours = self.parser.numofneighbours
        self.recommend_top_n = self.parser.recommend_top_n

    def process_test_key(self, test_key, train_dataset, test_projects):
        if not os.path.exists(train_dataset + '/' + self.parser.dict_path + '/dict__' + test_projects[test_key] + '.txt'):
            return

        recommendation = dict()
        test_pro = test_projects[test_key]

        if os.path.exists(train_dataset + '/' + self.parser.recommendation_path + '/' + test_pro + '.txt'):
            return

        lib_set = []

        user_item_matrix = self.build_user_item_matrix(train_dataset, test_pro, lib_set)
        similarities = get_similarity_matrix(train_dataset + '/' + self.parser.similarities_path + '/' + test_pro + '.txt',
                                             self.num_neighbours)

        val1 = np.sum(similarities[:self.num_neighbours])

        N = self.num_cols

        avg_rating = 1.0

        for i in range(N):
            if user_item_matrix[self.num_neighbours][i] == -1.0:
                val2 = 0.0
                for j in range(self.num_neighbours):
                    temp_rating = np.mean((user_item_matrix[j][0:N]))

                    val2 += (user_item_matrix[j][i] - temp_rating) * similarities[j]

                if val1 == 0.0:
                    recommendation[i] = avg_rating
                else:
                    recommendation[i] = (avg_rating + val2/val1)

        recommendation_result = sorted(recommendation.items(), key=lambda b: b[1], reverse=True)[:20]
        with open(file=train_dataset + '/' + self.parser.recommendation_path + '/' + test_pro + '.txt', mode='w') as fp:
            for key, val in recommendation_result:
                content = lib_set[key] + '\t' + str(val) + '\n'
                fp.write(content)

    def user_based_recommendation(self, train_dataset):
        test_projects = read_test_project(train_dataset + '/test_info.json')
        test_keys = test_projects.keys()

        recom_prog_bar = tqdm(desc='recommend progress',
                              leave=True,
                              total=len(test_keys))

        # 使用线程池并行处理每个测试项目
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_test_key, test_key, train_dataset, test_projects) for test_key in test_keys]

            # 等待所有线程完成
            for future in tqdm(futures, total=len(futures), desc='Thread progress', leave=False):
                future.result()
                recom_prog_bar.update()

        recom_prog_bar.close()
        print('recommend complete')

    def build_user_item_matrix(self, train_dataset, file_name, lib_set):
        # 保持原来的方法体不变
        # positive libs
        test_libs = get_train_libraries(train_dataset + '/' + self.parser.dict_path + '/dict__' + file_name + '.txt')
        sim_projects, all_neighbours, libraries = get_similarity_projects(train_dataset, file_name, self.parser)

        self.num_neighbours = len(sim_projects)
        self.parser.numofneighbours = self.num_neighbours

        all_neighbours[self.num_neighbours] = test_libs
        libraries = libraries | test_libs

        lib_set.extend(libraries)  # Using extend instead of a loop

        self.num_rows = self.num_neighbours + 1
        self.num_cols = len(libraries)

        # Initialize user_item_matrix as a NumPy array
        user_item_matrix = np.zeros((self.num_rows, self.num_cols), dtype=np.float64)

        # Fill user_item_matrix using vectorized operations
        for i in range(self.num_neighbours):
            tmp_libs = all_neighbours[i]
            for j, lib in enumerate(lib_set):
                if lib in tmp_libs:
                    user_item_matrix[i, j] = 1.0
                else:
                    user_item_matrix[i, j] = 0.0

        # Fill the last row with -1.0
        user_item_matrix[self.num_neighbours, :] = -1.0

        return user_item_matrix


if __name__ == '__main__':
    r_e = Recommendation()
    r_e.user_based_recommendation('./training dataset/' + r_e.parser.dataset)
