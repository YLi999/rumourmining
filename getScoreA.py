import json
import os.path
import sys

def getScoreA(reference_file,submission_file):
    truth_values = json.load(open(reference_file, 'r'))
    submission = json.load(open(submission_file, 'r'))

    observed = 0
    correct = 0

    #print(len(truth_values), 'entries in reference file')

    for reference_id in truth_values.keys():
        if reference_id in submission.keys():
            observed += 1
            if submission[reference_id] == truth_values[reference_id]:
                correct += 1
        else:
            print('unmatched entry:', reference_id, '-- no reference value for this document')

    score = correct / observed
    
    return score