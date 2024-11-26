import numpy as np
import matplotlib.pyplot as plt
from arguments import args

# Hyperparameters
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MOMENTUM = args.momentum

BASE_LRS = [1e-04, 1e-04, 1e-05]
WEIGHT_DECAYS = [1e-04, 1e-02, 1e-01]
RUNS = [1]
STAGE_1_EPOCHS = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
UPSAMPLES = [20, 50, 100]
valid_avg_accuracy_threshold_lower = 93
valid_avg_accuracy_threshold_upper = valid_avg_accuracy_threshold_lower + 1
worst_group_accuracy_list = []

for BASE_LR, WEIGHT_DECAY in zip(BASE_LRS, WEIGHT_DECAYS):
    for RUN in RUNS:
        for STAGE_1_EPOCH in STAGE_1_EPOCHS:
            for UPSAMPLE in UPSAMPLES: 
                STAGE_2_PATH = 'saved_models/run_%s/bs_%s_epochs_%s_lr_%s_wd_%s/epoch_%s/stage_2/incorrect_upsampled_%s_bs_%s_epochs_%s_lr_%s_wd_%s/'%(RUN, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY, STAGE_1_EPOCH, UPSAMPLE, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY)
                
                valid_avg_acc = np.loadtxt(STAGE_2_PATH + "plots_actual/valid_avg_acc.txt")
                valid_blond_men_acc = np.loadtxt(STAGE_2_PATH + "plots_actual/valid_blond_men_acc.txt")
                valid_blond_women_acc = np.loadtxt(STAGE_2_PATH + "plots_actual/valid_blond_women_acc.txt")
                valid_brunette_men_acc = np.loadtxt(STAGE_2_PATH + "plots_actual/valid_brunette_men_acc.txt")
                valid_brunette_women_acc = np.loadtxt(STAGE_2_PATH + "plots_actual/valid_brunette_women_acc.txt")
                valid_worst_group_acc = [min(a, b, c, d) for a,b,c,d in zip(valid_blond_men_acc, valid_blond_women_acc, valid_brunette_men_acc, valid_brunette_women_acc)]

                test_avg_acc = np.loadtxt(STAGE_2_PATH + "plots_actual/test_avg_acc.txt")
                test_blond_men_acc = np.loadtxt(STAGE_2_PATH + "plots_actual/test_blond_men_acc.txt")
                test_blond_women_acc = np.loadtxt(STAGE_2_PATH + "plots_actual/test_blond_women_acc.txt")
                test_brunette_men_acc = np.loadtxt(STAGE_2_PATH + "plots_actual/test_brunette_men_acc.txt")
                test_brunette_women_acc = np.loadtxt(STAGE_2_PATH + "plots_actual/test_brunette_women_acc.txt")
                test_worst_group_acc = [min(a, b, c, d) for a,b,c,d in zip(test_blond_men_acc, test_blond_women_acc, test_brunette_men_acc, test_brunette_women_acc)]

                correct_accuracies_blond = np.loadtxt(STAGE_2_PATH + "plots_actual/correct_accuracies_blond.txt")
                incorrect_accuracies_blond = np.loadtxt(STAGE_2_PATH + "plots_actual/incorrect_accuracies_blond.txt")
                estimated_worst_group_acc_blond = [min(x, y) for x,y in zip(correct_accuracies_blond, incorrect_accuracies_blond)]
                correct_accuracies_brunette = np.loadtxt(STAGE_2_PATH + "plots_actual/correct_accuracies_brunette.txt")
                incorrect_accuracies_brunette = np.loadtxt(STAGE_2_PATH + "plots_actual/incorrect_accuracies_brunette.txt")
                estimated_worst_group_acc_brunette = [min(x, y) for x,y in zip(correct_accuracies_brunette, incorrect_accuracies_brunette)]
                estimated_worst_group_acc = [min(a, b, c, d) for a,b,c,d in zip(correct_accuracies_blond, incorrect_accuracies_blond, correct_accuracies_brunette, incorrect_accuracies_brunette)]

                worst_group_accuracy = list(zip(list(range(1, 51)), 
                                             estimated_worst_group_acc_blond,
                                             estimated_worst_group_acc_brunette,
                                             estimated_worst_group_acc,
                                             valid_avg_acc, 
                                             valid_worst_group_acc,
                                             test_avg_acc, 
                                             test_worst_group_acc,
                                             [STAGE_1_EPOCH]*len(incorrect_accuracies_blond),
                                             [UPSAMPLE]*len(incorrect_accuracies_blond),
                                             [BASE_LR]*len(incorrect_accuracies_blond),
                                             [WEIGHT_DECAY]*len(incorrect_accuracies_blond)))
                worst_group_accuracy = [a for a in worst_group_accuracy if a[4] >= valid_avg_accuracy_threshold_lower and a[4] < valid_avg_accuracy_threshold_upper]
                worst_group_accuracy_list += worst_group_accuracy

estimated_worst_group_accuracy_list_sorted = sorted(worst_group_accuracy_list, key=lambda ele : ele[3])
estimated_worst_group_accuracy = estimated_worst_group_accuracy_list_sorted[-1]
estimated_test_avg_acc = estimated_worst_group_accuracy[6]
estimated_test_worst_group_accuracy = estimated_worst_group_accuracy[7]
print('Estimated Test Average Accuracy: ', estimated_test_avg_acc)
print('Estimated Worst-group Accuracy: ', estimated_test_worst_group_accuracy)

ground_truth_worst_group_accuracy_list_sorted = sorted(worst_group_accuracy_list, key=lambda ele : ele[5])
ground_truth_worst_group_accuracy = ground_truth_worst_group_accuracy_list_sorted[-1]
ground_truth_test_avg_acc = ground_truth_worst_group_accuracy[6]
ground_truth_test_worst_group_accuracy = ground_truth_worst_group_accuracy[7]
print('Ground Truth Test Average Accuracy: ', ground_truth_test_avg_acc)
print('Ground Truth Worst-group Accuracy: ', ground_truth_test_worst_group_accuracy)

