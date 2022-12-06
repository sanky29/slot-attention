import numpy as np

def micro(results, total):
    tp = 0
    for i in range(results.shape[0]):
        tp += results[i,i]
    recall = tp / total
    precision = recall
    return 2*(precision*recall)/(precision+recall)

def macro(results):
    tp_fp = np.sum(results, axis = 0)
    tp_fn = np.sum(results, axis = 1)
    classes = results.shape[0]
    precision = np.zeros(classes)
    recall = np.zeros(classes)
    for i in range(classes):
        precision[i] = results[i][i] / tp_fp[i]
        recall[i] = results[i][i] / tp_fn[i]
    f1 = 2*precision*recall/(precision+recall)
    p = sum(precision)/classes
    r = sum(recall)/classes
    return 2*(r*p)/(r+p), precision, recall, f1


def eval(true_y, pred_y, num_class = 3):
    
    label_to_id = dict()
    id_to_label = []
    results = np.zeros((num_class,num_class))

    total_examples = len(true_y)
    if(len(pred_y) != total_examples):
        print("RESULTS NOT SAME")
    
    for i in range(len(true_y)):
        if(true_y[i] not in label_to_id):
            label_to_id[true_y[i]] = len(label_to_id)
            id_to_label.append(true_y[i])
        if(pred_y[i] not in label_to_id):
            id_to_label.append(pred_y[i])
            label_to_id[pred_y[i]] = len(label_to_id)
        results[label_to_id[true_y[i]], label_to_id[pred_y[i]]] += 1
    r = dict()
    mic = micro(results, total_examples)
    macro_f1, class_p, class_r, class_f1 = macro(results)
    r['micro-f1'] = mic
    r['macro-f1'] = macro_f1
    r['class'] = dict()
    for i in range(len(label_to_id)):
        r['class'][id_to_label[i]] = dict()
        r['class'][id_to_label[i]]['f1'] = class_f1[i]
        r['class'][id_to_label[i]]['recall'] = class_r[i]
        r['class'][id_to_label[i]]['precision'] = class_p[i]

    return r
    

def eval_from_file(true_file, pred_file):
    pass

#checked from here
#https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f

# a = ['a','c','c','c','c','a','b','c','a','c']
# p = ['a','b','c','c','b','b','b','a','a','c']

# print(eval(a,p))