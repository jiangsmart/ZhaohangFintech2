import data_helper as d
import process as p
import datetime

if __name__ == '__main__':
    test_option = True
    if test_option:
        print "THIS IS A TEST"
    else:
        print "WARING: THIS IS A REAL WAR"

    train_path = './data/train_data'
    test_path = './data/test_data.csv'
    preprocess_path = './data/train_data_preprocess'
    out_path = './data/out.csv'
    timestamp = datetime.datetime.now()
    print "begin:\t\t\t\t\t" + timestamp.strftime('%Y.%m.%d-%H:%M:%S')

    corpus, test_corpus, old_ids = d.data_read2(test_option, train_path, test_path, preprocess_path)
    # print old_ids
    timestamp = datetime.datetime.now()
    print "data process done:\t\t" + timestamp.strftime('%Y.%m.%d-%H:%M:%S')

    sim = p.fun_bm25(corpus, test_corpus)
    result = p.resbysim(sim, old_ids)
    timestamp = datetime.datetime.now()
    print "result calculate done:\t" + timestamp.strftime('%Y.%m.%d-%H:%M:%S')

    d.data_write(result, out_path)
    timestamp = datetime.datetime.now()
    print "all done:\t\t\t\t" + timestamp.strftime('%Y.%m.%d-%H:%M:%S')
