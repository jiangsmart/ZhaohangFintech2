# coding=utf-8
in_path = './data/out.csv'
test_path = './data/test_data.csv'
out_path = './data/sample_final3.txt'
train_path = './data/train_data.csv'
watch_path = './data/watch_final3.txt'

file_test = open(test_path)
lines = file_test.readlines()
file_test.close()

train_file = open(train_path)

origin = train_file.readline()
origin = origin.split('\r')
train_file.close()
corpus = []
for line in origin:
    corpus.append(line.split(',')[1])

ids = []

for i, line in enumerate(lines):
    # 第一行是表头
    if i == 0:
        continue
    test_id = int(line.strip().split(',')[0])
    ids.append(test_id)
print ids
print len(ids)

in_file = open(in_path)
lines = in_file.readlines()
in_file.close()

watch_file = open(watch_path, 'w')
out_file = open(out_path, 'w')
out_file.write('source_id	target_id	similarity(可选)	source_title(可选)	target_title(可选)\n')
for i, test_id in enumerate(ids):
    count = 0
    for result_id in map(int, lines[i].split(',')):
        if test_id == result_id:
            continue
        out_file.write(str(test_id) + '\t' + str(result_id) + '\n')
        count += 1
        watch_file.write(
            '*************' + str(count) + '*************\n' + corpus[test_id] + '\n' + corpus[result_id] + '\n')
        if count == 20:
            break

out_file.close()
watch_file.close()
