# coding=utf-8
import process as p
import codecs


def data_read2(test_option, train_path, test_path, preprocess_path):
    if test_option:
        test_str = "_test"
    else:
        test_str = ""

    # stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    # stop_flag = ['x']
    stop_flag = []
    stop_words = codecs.open('./data/stopwords.dat', 'r', encoding='utf8').readlines()
    stop_words = [w.strip() for w in stop_words]
    # stop_words = []

    file_train = open(train_path + test_str + '.csv')
    if test_option:
        lines = file_train.readlines()
    else:
        all_file = file_train.readline()
        lines = all_file.split('\r')
    file_train.close()

    print 'train_num before filter:', len(lines)
    old_ids, lines = p.quchong(lines)  # 通过查询old_ids可以得到原来的id，例如ids[3]=5
    print 'train_num after filter:', len(lines)

    file_preprocess = open(preprocess_path + test_str + '.csv', 'w')
    contents = []
    for i, content in enumerate(lines):
        if i % 1000 == 0:
            print i
        content = p.preprocess3(content, stop_flag, stop_words)
        contents.append(content)
        file_preprocess.write(str(old_ids[i]) + ',' + " ".join(content).encode('utf-8') + '\n')
    file_preprocess.close()

    if test_option:
        lines = ['id', '1,', '4,', '5,']
    else:
        file_test = open(test_path)
        lines = file_test.readlines()
        file_test.close()

    test_contents = []
    for i, line in enumerate(lines):
        # 第一行是表头
        if i == 0:
            continue
        ii = int(line.strip().split(',')[0])
        ii = p.findnowid(ii, old_ids)
        test_content = contents[ii]
        test_contents.append(test_content)

    return contents, test_contents, old_ids


def data_write(result, out_path):
    # out_path = './data/out.csv'
    file_out = open(out_path, 'w')
    for r in result:
        file_out.write(','.join(map(str, r)))
        file_out.write('\n')
    file_out.close()
    return


if __name__ == '__main__':
    pass
    # stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    # stop_words = codecs.open('./data/stopwords.dat', 'r', encoding='utf8').readlines()
    # stop_words = [w.strip() for w in stop_words]
    # content = ""
    # content = p.preprocess2(content, stop_flag, stop_words)
