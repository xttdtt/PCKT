import os
import time
import pandas as pd
import numpy as np
from scipy import sparse
from collections import defaultdict


# Clean up answering time data that is more than 2 standard deviations from the mean
def clean_abnormal(num, mean, std):
    min_normal = mean - 2 * std
    max_normal = mean + 2 * std
    if num > max_normal or num < min_normal:
        return 1
    return 0


# Calculate similarity
def similarity(set1, set2):
    union = set(set1).union(set(set2))
    intersection = set(set1).intersection(set(set2))
    return (len(intersection) / len(union))


# Save data file
def save_dict(dict_name, file_name):
    with open(file_name, 'w') as f:
        f.write(str(dict_name))


# Process dataset and clean up abnormal data
def process_data(dataset, datafolder, pre_file, post_file, cols):
    starttime = time.time()
    df = pd.read_csv(os.path.join(dataset, datafolder, pre_file), encoding='ISO-8859-1', low_memory=False)
    df = df.dropna(subset=['skill_id'])
    df = df[df['original'].isin([1])]
    df = df.dropna(subset=['ms_first_response'])
    df = df[df["ms_first_response"] > 0]
    students = df.groupby(['user_id'], as_index=True)
    delete_students = []
    for student in students:
        if len(student[1]) < 5:
            delete_students.append(student[0])
    df = df[~df['user_id'].isin(delete_students)]
    # Extract the required column information
    df = df[cols]
    # Clean up abnormal data
    problems = df['problem_id'].unique()
    delete_lines = []
    for pro_id in range(len(problems)):
        # Find the row data corresponding to each question
        tmp_df = df[df['problem_id'] == problems[pro_id]]
        # Find the row number of the row data
        tmp_lines = tmp_df.index
        # Calculate the mean and standard deviation
        mean_ms, std_ms = tmp_df["ms_first_response"].mean(), tmp_df["ms_first_response"].std()
        for line in tmp_lines:
            # Take out the answering time data of the row data
            tmp_ms = tmp_df[tmp_df.index == line]["ms_first_response"].values
            # Clean up if it is abnormal data
            if (clean_abnormal(tmp_ms, mean_ms, std_ms)):
                delete_lines.append(line)
    df = df[~df.index.isin(delete_lines)]
    # Answering time data changed from milliseconds to seconds
    df["ms_first_response"] /= 1000
    df.to_csv(os.path.join(dataset, datafolder, post_file))
    endtime = time.time()
    print("process_data time:", endtime - starttime)


# Extract problems and students and their corresponding ids
def extract_pro_stu_id(dataset, datafolder, post_file):
    starttime = time.time()
    df = pd.read_csv(os.path.join(dataset, datafolder, post_file), encoding="ISO-8859-1", low_memory=True)
    problems, students = df['problem_id'].unique(), df['user_id'].unique()
    num_pro, num_stu = len(problems), len(students)
    pro_id_dict, stu_id_dict = dict(zip(problems, range(num_pro))), dict(zip(students, range(num_stu)))
    save_dict(num_pro, os.path.join(dataset, datafolder, 'num_pro.txt'))
    save_dict(num_stu, os.path.join(dataset, datafolder, 'num_stu.txt'))
    save_dict(pro_id_dict, os.path.join(dataset, datafolder, 'pro_id_dict.txt'))
    save_dict(stu_id_dict, os.path.join(dataset, datafolder, 'stu_id_dict.txt'))
    endtime = time.time()
    print("extract_pro_stu_id time:", endtime - starttime)
    return df, num_pro, num_stu, problems, students


# Extraction of problem-skill relationship data
def extract_pro_skill(dataset, datafolder, df, num_pro, problems):
    starttime = time.time()
    pro_skill_adj, skill_id_dict, pro_skill_dict = [], {}, defaultdict(list)
    count_skill, max_skill_len = 0, -np.inf
    for pro_id in range(num_pro):
        # Find all rows of data corresponding to each question
        tmp_df = df[df['problem_id'] == problems[pro_id]]
        # Just add the corresponding skill to any row of data for each question, here we choose the first row
        tmp_skills = [ele for ele in tmp_df.iloc[0]['skill_id'].split('_')]
        # Find the maximum number of skills in question
        max_skill_len = max(len(tmp_skills), max_skill_len)
        for skill in tmp_skills:
            if skill not in skill_id_dict:
                skill_id_dict[skill] = count_skill
                count_skill += 1
            pro_skill_adj.append([pro_id, skill_id_dict[skill], 1])
            pro_skill_dict[pro_id].append(skill_id_dict[skill])
    num_skill = len(skill_id_dict)
    print('max skill number %d' % (max_skill_len))
    save_dict(max_skill_len, os.path.join(dataset, datafolder, 'max_skill_len.txt'))
    save_dict(num_skill, os.path.join(dataset, datafolder, 'num_skill.txt'))
    save_dict(skill_id_dict, os.path.join(dataset, datafolder, 'skill_id_dict.txt'))
    save_dict(dict(pro_skill_dict), os.path.join(dataset, datafolder, 'pro_skill_dict.txt'))
    pro_skill_adj = np.array(pro_skill_adj).astype(np.int32)
    pro_skill_sparse = sparse.coo_matrix((pro_skill_adj[:, 2].astype(np.float32), (pro_skill_adj[:, 0], pro_skill_adj[:, 1])), shape=(num_pro, num_skill))
    sparse.save_npz(os.path.join(dataset, datafolder, 'pro_skill_sparse.npz'), pro_skill_sparse)
    endtime = time.time()
    print("extract_pro_skill time:", endtime - starttime)
    return num_skill


# Extracting problem difficulty data
def extract_pro_diff(dataset, datafolder, df, num_pro, problems):
    starttime = time.time()
    pro_diff_adj = []
    for pro_id in range(num_pro):
        # Find all rows of data corresponding to each question
        tmp_df = df[df['problem_id'] == problems[pro_id]]
        # Find the rows of data in each question row that the student answered correctly
        tmp_df_corr = tmp_df[tmp_df["correct"] == 1]
        # Calculate the average time to answer each question correctly (representing the speed of answering the problems)
        tmp_time_pro_corr = 0
        if len(tmp_df_corr):
            tmp_time_pro_corr = tmp_df_corr["ms_first_response"].mean()
        # Calculate the percentage of correct answers for each question (representing the accuracy of answering the problems)
        tmp_acc_pro_corr = len(tmp_df_corr) / len(tmp_df)
        tmp_pro_diff_adj = [0.] * 3
        tmp_pro_diff_adj[0], tmp_pro_diff_adj[1] = tmp_time_pro_corr, tmp_acc_pro_corr
        pro_diff_adj.append(tmp_pro_diff_adj)
    # Answering time data normalization
    pro_diff_adj = np.array(pro_diff_adj).astype(np.float32)
    pro_diff_adj[:, 0] = (pro_diff_adj[:, 0] - np.min(pro_diff_adj[:, 0])) / (np.max(pro_diff_adj[:, 0]) - np.min(pro_diff_adj[:, 0]))
    # Problem difficulty = accuracy/speed
    pro_diff_adj[:, 2] = pro_diff_adj[:, 1] / (pro_diff_adj[:, 0] + 1e-4)
    # Problem difficulty data normalization
    pro_diff_adj[:, 2] = (pro_diff_adj[:, 2] - np.min(pro_diff_adj[:, 2])) / (np.max(pro_diff_adj[:, 2]) - np.min(pro_diff_adj[:, 2]))
    pro_diff_list = pro_diff_adj[:, 2]
    pro_diff_sparse = sparse.coo_matrix(pro_diff_list, shape=(1, num_pro))
    sparse.save_npz(os.path.join(dataset, datafolder, 'pro_diff_sparse.npz'), pro_diff_sparse)
    endtime = time.time()
    print("extract_pro_diff time:", endtime - starttime)


# Extraction of data related to students and skills
def extract_stu_skill(dataset, datafolder, df, num_pro, num_stu, num_skill, problems):
    starttime = time.time()
    with open(os.path.join(dataset, datafolder, 'stu_id_dict.txt'), 'r') as f:
        stu_id_dict = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'skill_id_dict.txt'), 'r') as f:
        skill_id_dict = eval(f.read())
    stu_skill_total_adj, stu_skill_corr_adj, stu_skill_adj = np.zeros((num_stu, num_skill)), np.zeros((num_stu, num_skill)), np.zeros((num_stu, num_skill))
    for pro_id in range(num_pro):
        tmp_df = df[df['problem_id'] == problems[pro_id]]
        tmp_skills = [ele for ele in tmp_df.iloc[0]['skill_id'].split('_')]
        tmp_lines = tmp_df.index
        for line in tmp_lines:
            tmp_stu = int(tmp_df[tmp_df.index == line]["user_id"].values)
            tmp_stu_id = stu_id_dict[tmp_stu]
            # If the student answers the problem correctly, the student is considered to correctly answer the skill corresponding to the problem
            if tmp_df[tmp_df.index == line]["correct"].values == 1:
                for skill in tmp_skills:
                    stu_skill_corr_adj[tmp_stu_id][skill_id_dict[skill]] += 1
                    # Add 1 to the number of times students answer the skill, regardless of whether the student answer correctly or not
                    stu_skill_total_adj[tmp_stu_id][skill_id_dict[skill]] += 1
            else:
                for skill in tmp_skills:
                    stu_skill_total_adj[tmp_stu_id][skill_id_dict[skill]] += 1
    # Count the total number of answers for the skill
    skill_total_list = np.sum(stu_skill_total_adj, 0)
    # Count the total number of correct answers for the skill
    skill_corr_list = np.sum(stu_skill_corr_adj, 0)
    # Calculate the difficulty of the skill = total number of correct answers / total number of answers
    skill_diff_list = np.zeros(num_skill)
    for i in range(num_skill):
        if skill_total_list[i]:
            skill_diff_list[i] = skill_corr_list[i] / skill_total_list[i]
    # Calculation of students' mastery of skills = total number of student correct answers for skills / total number of student answers for skills
    for i in range(num_stu):
        for j in range(num_skill):
            if stu_skill_total_adj[i][j]:
                stu_skill_adj[i][j] = stu_skill_corr_adj[i][j] / stu_skill_total_adj[i][j]
    skill_diff_sparse = sparse.coo_matrix(skill_diff_list, shape=(1, num_skill))
    sparse.save_npz(os.path.join(dataset, datafolder, 'skill_diff_sparse.npz'), skill_diff_sparse)
    stu_skill_sparse = sparse.coo_matrix(stu_skill_adj, shape=(num_stu, num_skill))
    sparse.save_npz(os.path.join(dataset, datafolder, 'stu_skill_sparse.npz'), stu_skill_sparse)
    endtime = time.time()
    print("extract_stu_skill time:", endtime - starttime)


# Extraction of problem-problem, skill-skill relationship
def extract_pro_skill_similarity(dataset, datafolder):
    starttime = time.time()
    pro_skill_coo = sparse.load_npz(os.path.join(dataset, datafolder, "pro_skill_sparse.npz"))
    [num_pro, num_skill] = pro_skill_coo.toarray().shape
    pro_skill_csc = pro_skill_coo.tocsc()
    pro_skill_csr = pro_skill_coo.tocsr()
    """1. Extracting problem-problem relationships"""
    pro_pro_adj, temp_pro_pro_simi = [], []
    for pro_index in range(num_pro):
        # Take out the skills that the problem has
        tmp_skills = pro_skill_csr.getrow(pro_index).indices
        # Take the set of problems that have the same skill as the problem
        similar_pros = pro_skill_csc[:, tmp_skills].indices
        # Remove duplicate elements
        similar_pros = list(set(similar_pros))
        # Storage problem - problem relationship element location
        zipped = zip([pro_index] * len(similar_pros), similar_pros)
        pro_pro_adj += list(zipped)
        # Calculate the similarity of problems with the same skills
        for pro in similar_pros:
            # Take out skills related to problem sets
            tmp_pro_skills = pro_skill_csr.getrow(pro).indices
            # Calculate similarity
            temp_pro_pro_simi.append(similarity(tmp_skills, tmp_pro_skills))
    pro_pro_adj = np.array(pro_pro_adj).astype(np.int32)
    pro_pro_sparse = sparse.coo_matrix((temp_pro_pro_simi, (pro_pro_adj[:, 0], pro_pro_adj[:, 1])), shape=(num_pro, num_pro))
    sparse.save_npz(os.path.join(dataset, datafolder, 'pro_pro_sparse.npz'), pro_pro_sparse)
    """2. Extracting skill-skill relationships"""
    skill_skill_adj, temp_skill_skill_simi = [], []
    for skill_index in range(num_skill):
        tmp_pros = pro_skill_csc.getcol(skill_index).indices
        similar_skills = pro_skill_csr[tmp_pros, :].indices
        similar_skills = list(set(similar_skills))
        zipped = zip([skill_index] * len(similar_skills), similar_skills)
        skill_skill_adj += list(zipped)
        for skill in similar_skills:
            tmp_skill_pros = pro_skill_csc.getcol(skill).indices
            temp_skill_skill_simi.append(similarity(tmp_pros, tmp_skill_pros))
    skill_skill_adj = np.array(skill_skill_adj).astype(np.int32)
    skill_skill_sparse = sparse.coo_matrix((temp_skill_skill_simi, (skill_skill_adj[:, 0], skill_skill_adj[:, 1])), shape=(num_skill, num_skill))
    sparse.save_npz(os.path.join(dataset, datafolder, 'skill_skill_sparse.npz'), skill_skill_sparse)
    endtime = time.time()
    print("extract_pro_skill_similarity time:", endtime - starttime)


# Extract the students, problems, skills and answers in each row of data and save them as corresponding ids
def extract_stu_pro_skill_corr(dataset, datafolder, df):
    starttime = time.time()
    with open(os.path.join(dataset, datafolder, 'pro_id_dict.txt'), 'r') as f:
        pro_id_dict = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'pro_skill_dict.txt'), 'r') as f:
        pro_skill_dict = eval(f.read())
    with open(os.path.join(dataset, datafolder, 'stu_id_dict.txt'), 'r') as f:
        stu_id_dict = eval(f.read())
    df1 = df[["user_id", "problem_id", "skill_id", "correct"]]
    df1["problem_id"] = df1["problem_id"].map(lambda pro: pro_id_dict[pro])
    df1["skill_id"] = df1["problem_id"].map(lambda pro: pro_skill_dict[pro])
    df1["user_id"] = df1["user_id"].map(lambda stu: stu_id_dict[stu])

    stu_pro_skill_corr = []

    for line in df1.index:
        tmp_stu = int(df1[df1.index == line]["user_id"].values)
        tmp_pro = int(df1[df1.index == line]["problem_id"].values)
        tmp_skill = df1[df1.index == line]["skill_id"].values.tolist().pop(0)
        tmp_corr = int(df1[df1.index == line]["correct"].values)
        tmp_stu_pro_skill_corr = [tmp_stu, tmp_pro, tmp_skill, tmp_corr]
        stu_pro_skill_corr.append(tmp_stu_pro_skill_corr)
    np.savez(os.path.join(dataset, datafolder, 'stu_pro_skill_corr.npz'), stu_pro_skill_corr=stu_pro_skill_corr)
    endtime = time.time()
    print("extract_stu_pro_skill_corr time:", endtime - starttime)


if __name__ == '__main__':
    starttime = time.time()
    dataset = "Assist09"
    datafolder = "Data"
    pre_file = dataset + "_original.csv"
    post_file = dataset + ".csv"
    cols = ["user_id", "problem_id", "skill_id", "correct", "ms_first_response"]
    process_data(dataset, datafolder, pre_file, post_file, cols)
    df, num_pro, num_stu, problems, students = extract_pro_stu_id(dataset, datafolder, post_file)
    num_skill = extract_pro_skill(dataset, datafolder, df, num_pro, problems)
    print('dataset {0},problem number {1},student number {2},skill number {3}'.format(dataset, num_pro, num_stu, num_skill))
    extract_pro_diff(dataset, datafolder, df, num_pro, problems)
    extract_stu_skill(dataset, datafolder, df, num_pro, num_stu, num_skill, problems)
    extract_pro_skill_similarity(dataset, datafolder)
    extract_stu_pro_skill_corr(dataset, datafolder, df)
    endtime = time.time()
    print("total time :", endtime - starttime)
