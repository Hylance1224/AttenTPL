import os

from utility.DataReader import *
from utility.parser import *

from tqdm import tqdm


class Recommendation:
    def __init__(self):
        self.parser = arg_parse()
        self.num_cols = 0
        self.num_rows = 0
        self.num_neighbours = self.parser.numofneighbours
        self.recommend_top_n = self.parser.recommend_top_n

    def user_based_recommendation(self, train_dataset):

        test_projects = read_test_project(train_dataset + '/test_info.json')

        test_keys = test_projects.keys()

        recom_prog_bar = tqdm(desc='recommend progress',
                              leave=True,
                              total=len(test_keys))

        for test_key in test_keys:

            if not os.path.exists(train_dataset + '/' + self.parser.dict_path + '/dict__' + test_projects[test_key] + '.txt'):
                continue

            recommendation = dict()
            test_pro = test_projects[test_key]
            lib_set = []
            val1 = 0.0
            # 需要把这里改一下
            user_item_matrix = self.build_user_item_matrix(train_dataset, test_pro, lib_set)
            similarities = get_similarity_matrix(train_dataset + '/' + self.parser.similarities_path + '/' + test_pro + '.txt',
                                                 self.num_neighbours)

            for i in range(self.num_neighbours):
                val1 += similarities[i]
            N = self.num_cols

            avg_rating = 1.0

            for i in range(N):
                if user_item_matrix[self.num_neighbours][i] == -1.0:
                    val2 = 0.0
                    for j in range(self.num_neighbours):
                        temp_rating = 0.0
                        for k in range(N):
                            temp_rating += user_item_matrix[j][k]
                        temp_rating = temp_rating/N
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
            del user_item_matrix, similarities
            recom_prog_bar.update()

        recom_prog_bar.close()
        print('recommend complete')

    def build_user_item_matrix(self, train_dataset, file_name, lib_set):

        # positive libs
        test_libs = get_train_libraries(train_dataset + '/' + self.parser.dict_path + '/dict__' + file_name + '.txt')
        sim_projects, all_neighbours, libraries = get_similarity_projects(train_dataset, file_name, self.parser, test_libs)

        self.num_neighbours = len(sim_projects)
        self.parser.numofneighbours = self.num_neighbours

        all_neighbours[self.num_neighbours] = test_libs
        libraries = libraries | test_libs

        for lib in libraries:
            lib_set.append(lib)

        self.num_rows = self.num_neighbours + 1
        self.num_cols = len(libraries)

        user_item_matrix = []
        for i in range(self.num_rows):
            user_item_matrix.append([j for j in range(self.num_cols)])

        for i in range(self.num_neighbours):
            tmp_libs = all_neighbours[i]
            for j in range(self.num_cols):
                if lib_set[j] in tmp_libs:
                    user_item_matrix[i][j] = 1.0
                else:
                    user_item_matrix[i][j] = 0.

        # for j in range(self.num_cols):
        #     user_item_matrix[self.num_neighbours][j] = -1.0
        tmp_libs = all_neighbours[self.num_neighbours]
        for j in range(self.num_cols):
            str = lib_set[j]
            if str in tmp_libs:
                user_item_matrix[self.num_neighbours][j] = 1.0
            else:
                user_item_matrix[self.num_neighbours][j] = -1.0

        return user_item_matrix


if __name__ == '__main__':
    r_e = Recommendation()
    r_e.user_based_recommendation('./training dataset/' + r_e.parser.dataset)
