'''
Note: Code has been heavily adapted from Christopher V. Cosgriff, MD, MPH work on
sequential severity prediction for critically ill patients
(Source: https://github.com/cosgriffc/seq-severityscore)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns



# Functions for generating ROC curves and calculating AUROC
def auc_ci(f_hat, y_true, n_bootstraps=2000, ci_level=0.95):
    li = (1. - ci_level)/2
    ui = 1 - li

    rng = np.random.RandomState(seed=42)
    bootstrapped_auc = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(f_hat), len(f_hat))
        auc = roc_auc_score(y_true[indices], f_hat[indices])
        bootstrapped_auc.append(auc)

    sorted_scores = np.array(bootstrapped_auc)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(li * len(sorted_scores))]
    confidence_upper = sorted_scores[int(ui * len(sorted_scores))]

    return confidence_lower, confidence_upper

def gen_auc_plot(models, names, title, X, y, ci_level=None, save_name=None):
    plt.figure(figsize=(10, 10))
    for i, model in enumerate(models):
        f_hat = model.predict_proba(X)
        roc = roc_curve(y, f_hat[:, 1])
        auc = roc_auc_score(y, f_hat[:, 1])
        if (ci_level != None):
            ci = auc_ci(f_hat=f_hat[:, 1], y_true=y, ci_level=ci_level)
            sns.lineplot(x=roc[0], y=roc[1], label='{0}\n(AUC = {1:.3f} [{2:.3f}, {3:.3f}])'.format(names[i], auc, *ci))
        else:
            sns.lineplot(x=roc[0], y=roc[1], label='{0}\n(AUC = {1:.3f})'.format(names[i], auc))
    plt.plot([0, 1], [0, 1], 'k:')
    plt.xlabel('1 - Specificty')
    plt.ylabel('Sensitivity')
    plt.title(title)
    if save_name != None:
        plt.savefig(save_name + '.jpg', bbox_inches='tight')
    plt.show()

# Functions for generating PRC curves and calculating AUPRC (AP)
def prc_ci(f_hat, y_true, n_bootstraps=2000, ci_level=0.95):
    li = (1. - ci_level)/2
    ui = 1 - li

    rng = np.random.RandomState(seed=42)
    bootstrapped_ap = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(f_hat), len(f_hat))
        ap = average_precision_score(y_true[indices], f_hat[indices])
        bootstrapped_ap.append(ap)

    sorted_scores = np.array(bootstrapped_ap)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(li * len(sorted_scores))]
    confidence_upper = sorted_scores[int(ui * len(sorted_scores))]

    return confidence_lower, confidence_upper

def gen_prc_plot(models, names, title, X, y, ci_level=None, save_name=None):
    plt.figure(figsize=(10, 10))
    for i, model in enumerate(models):
        f_hat = model.predict_proba(X)
        precision, recall, _ = precision_recall_curve(y, f_hat[:, 1])
        ap = average_precision_score(y, f_hat[:, 1])
        if ci_level != None:
            ci = prc_ci(f_hat=f_hat[:, 1], y_true=y, ci_level=ci_level)
            sns.lineplot(x=recall, y=precision, label='{0}\n(AP = {1:.3f} [{2:.3f}, {3:.3f}])'.format(names[i], ap, *ci))
        else:
            sns.lineplot(x=recall, y=precision, label='{0}\n(AP = {1:.3f})'.format(names[i], ap))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    if save_name != None:
        plt.savefig(save_name + '.jpg', bbox_inches='tight')
    plt.show()

# Function for generating reliability curves
def gen_calib_plot(models, names, title, X, y, save_name=None):
    plt.figure(figsize=(10, 10))
    for i, model in enumerate(models):
        f_hat = model.predict_proba(X)
        fraction_of_positives, mean_predicted_value = calibration_curve(y, f_hat[:, 1], n_bins=5)
        sns.lineplot(x=mean_predicted_value, y=fraction_of_positives, label=names[i])
    plt.plot([0, 1], [0, 1], 'k:')
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Actual Probability of Outcome')
    plt.title(title)
    if save_name != None:
        plt.savefig(save_name + '.jpg', bbox_inches='tight')
    plt.show()

# Functions for generating NRI
def nri(model1, model2, X, y):
    n = y.shape[0]
    n_event = y.sum()
    n_no_event = n - n_event

    pred_model1 = model1.predict_proba(X)[:, 1]
    pred_model2 = model2.predict_proba(X)[:, 1]
    z_pos = (pred_model2[y == 1] > pred_model1[y == 1]).sum() - (pred_model2[y == 1] < pred_model1[y == 1]).sum()
    z_neg = (pred_model2[y == 0] < pred_model1[y == 0]).sum() - (pred_model2[y == 0] > pred_model1[y == 0]).sum()

    additive_nri = (z_pos/n_event)*100 + (z_neg/n_no_event)*100
    absolute_nri = (z_pos + z_neg)/n
    return (additive_nri, absolute_nri)

def nri_grid(models, names, X, y, round=True):
	n_models = len(models)
	additive_grid = np.zeros(shape=(n_models, n_models), dtype=float)
	absolute_grid = np.zeros(shape=(n_models, n_models), dtype=float)
	for i, model1 in enumerate(models):
		for j, model2 in enumerate(models):
			if (i < j):
				additive_nri, absolute_nri = nri(model1=model1, model2=model2, X=X, y=y)
				additive_grid[i, j] = additive_nri
				absolute_grid[i, j] = absolute_nri
	additive_grid = additive_grid + np.flip(additive_grid)*-1
	absolute_grid = absolute_grid + np.flip(absolute_grid)*-1
	additive_grid = pd.DataFrame(data=additive_grid, index=names, columns=names)
	absolute_grid = pd.DataFrame(data=absolute_grid, index=names, columns=names)
	if round:
		return (additive_grid.round(0), absolute_grid.round(2))
	else:
		return (additive_grid, absolute_grid)

# Functions for generating OPR table
def op_ratio_ci(f_hat, y_true, n_bootstraps=2000, ci_level=0.95):
    li = (1. - ci_level)/2
    ui = 1. - li

    rng = np.random.RandomState(seed=42)
    bootstrapped_opr = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(f_hat), len(f_hat))
        opr = y_true[indices].mean() / f_hat[indices].mean()
        bootstrapped_opr.append(opr)

    sorted_scores = np.array(bootstrapped_opr)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(li * len(sorted_scores))]
    confidence_upper = sorted_scores[int(ui * len(sorted_scores))]

    return confidence_lower, confidence_upper

def op_ratio(model, X, y):
    try:
        f_hat = model.predict_proba(X)[:,1]
    except:
        f_hat = model.predict_proba(X)

    observed = y.mean()
    predicted = f_hat.mean()
    return (observed / predicted, *op_ratio_ci(f_hat, y))

def opr_table(models, names, X, y):
	opr = [op_ratio(model, X, y) for model in models]
	opr_table = pd.DataFrame(data=opr, index=names, columns=['OPR', '2.5%', '97.5%'])
	return opr_table

# Feature analysis function for normalized regression model
def gen_logodds_plot(model, features, n_features=10, title='Log Odds Plot', save_name=None):
    plt.figure(figsize=(10, 10))
    coef = {k:v for k, v in zip(features, model.named_steps['ridge'].coef_.ravel())}
    coef = pd.DataFrame.from_dict(coef, orient='index', columns=['log_odds'])
    coef = coef.reindex(coef.log_odds.abs().sort_values(ascending=False).index).iloc[0:n_features, :]
    coef = coef.reindex(index=coef.index[::-1])
    pos_neg = coef.log_odds > 0
    color_map = pos_neg.map({True: tableau20[8], False: tableau20[10]})
    coef.plot(kind='barh', grid=True, sort_columns=False,
                   title=title,
                   color=[color_map.values], ax=plt.axes(), width=0.20,
                   legend=False)
    plt.xlabel('log(OR)')
    plt.ylabel('Features')
    if save_name != None:
        plt.savefig('./figures/' + save_name + '.jpg', bbox_inches='tight')
    plt.show()


# Functions for getting specificity at given sensitivity
def specf_ci(f_hat, y_true, sens, n_bootstraps=500, ci_level=0.95):
    li = (1. - ci_level)/2
    ui = 1 - li

    rng = np.random.RandomState(seed=42)
    bootstrapped_test_stat = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(f_hat), len(f_hat))
        threshold = get_threshold_for_sens(f_hat[indices],y_true[indices],sens)
        sensitivity, specificity = get_sens_spec(f_hat[indices],y_true[indices],threshold)
        bootstrapped_test_stat.append(specificity)

    sorted_scores = np.array(bootstrapped_test_stat)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(li * len(sorted_scores))]
    confidence_upper = sorted_scores[int(ui * len(sorted_scores))]

    return confidence_lower, confidence_upper

def get_sens_spec(f_hat,y_true,threshold):
    tn, fp, fn, tp = confusion_matrix(y_true,f_hat>threshold).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    return sensitivity, specificity

def scan_thresold(f_hat,y_true,low_threshold,high_threshold,step,sens):
    threshold_values = np.arange(low_threshold,high_threshold,step)[::-1]

    for threshold in threshold_values:
        sensitivity, specificity = get_sens_spec(f_hat,y_true,threshold)
        if specificity > 0:
            if sensitivity >= sens:
                return threshold
    return low_threshold

def get_threshold_for_sens(f_hat,y_true,sens):
    high_threshold = 0.5
    low_threshold = 0.01
    step = 0.1
    for i in range(4):
        low_threshold = scan_thresold(f_hat,y_true,low_threshold,high_threshold,step,sens)
        high_threshold = low_threshold + step
        step = step/10.0
    return low_threshold

def get_spec_for_sens(model, X, y,sens):
    try:
        f_hat = model.predict_proba(X)[:,1]
    except:
        f_hat = model.predict_proba(X)

    threshold = get_threshold_for_sens(f_hat,y,sens)
    # Sensitivity and specificity
    sensitivity, specificity = get_sens_spec(f_hat,y,threshold)
    return (sensitivity , specificity, *specf_ci(f_hat, y, sens))

def sens_spec_table(models, names, X, y,sens=0.995):
	ss = [get_spec_for_sens(model, X, y,sens) for model in models]
	ss_table = pd.DataFrame(data=ss, index=names, columns=['Sens','Spec', '2.5%', '97.5%'])
	return ss_table

# Wrapper for working with APACHE model from eICU which just provides
# the probabilities from the model.
class APACHEWrapper:
    def __init__(self, pos_preds_proba):
        self.proba = pos_preds_proba

    def predict_proba(self, X):
        return np.array([1-self.proba, self.proba]).T

class SVCWrapper:
    def __init__(self,model):
        self.model = model

    def predict_proba(self,X):
        return self.model.decision_function(X)
