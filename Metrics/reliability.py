
import Utils as ut
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error



        
    
#se entrambi hanno torto/ragione (o a prescindere), quanto sono convinti della ground truth?
#rende un'array (n.instances, 2) una colonna di confidence per modello se in input ha una lista di due modelli
#rende l'array delle probs sulla gt se in input ha un solo modello
#concordant può essere True/False/None
def true_class_probability(models, X_ts, y_true, concordant = None, **kwargs):

    _, probs = ut.predictions(models, X_ts)
    if isinstance(models, (list, tuple)) and len(models) > 1:
    
        if concordant is not None:
            probs[0], probs[1], y_true = ut.prediction_concordance_filter(models, X_ts, to_filter = (probs[0], probs[1], y_true), y_true = y_true, concordant = concordant)
        
        indexes = y_true.ravel().astype(int)
        probs_true_class = [prob[np.arange(prob.shape[0]), indexes] for prob in probs]
        return np.column_stack((probs_true_class[0], probs_true_class[1])) 
    else:
        probs = (probs)
        indexes = y_true.ravel().astype(int)
        prob_true_class = probs[np.arange(probs.shape[0]), indexes]        
        return prob_true_class 


#calcola l'errore di calibrazione atteso del modello
#sia per cl binaria che multiclasse perché considera a prescindere la max confidence
def expected_calibration_error(model, X_ts, y_true, n_bins = 5, **kwargs):
    y_pred, probs = ut.predictions(model, X_ts)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    max_conf = np.max(probs, axis = 1)
    accuracies = y_pred == y_true
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(max_conf > bin_lower.item(), max_conf <= bin_upper.item())
        prob_in_bin = in_bin.mean() #prob che la confidence sia nel bin

        if prob_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_conf_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_conf_in_bin - accuracy_in_bin) * prob_in_bin
    return ece



#calcola la sicurezza media sulla ground truth
def mean_confidence_on_ground_truth(model, X_ts, y_true, **kwargs):
    probs_of_gt = true_class_probability(model, X_ts, y_true)
    return np.mean(probs_of_gt)


#calcola la media dei logaritmi della sicurezza sulla gt
#è più sensibile della sicurezza media rispetto alle probabilità molto basse: ne tiene conto
def cat_cross_entropy_on_ground_truth(model, X_ts, y_true, **kwargs):
    probs_of_gt = true_class_probability(model, X_ts, y_true)
    neg_logs = - np.log(probs_of_gt + 1e-15) #per evitare log(0)
    return np.mean(neg_logs)

#calcola la distanza vettoriale o correlazione tra i vettori delle confidence sulla ground truth
#rende uno scalare, che sarebbe la similarità/distanza
def true_class_prob_similarity(models, X_ts, y_true, metric, concordant = False, **kwargs):
    true_class_probs = true_class_probability(models, X_ts, y_true, concordant = concordant)
    
    p1 = true_class_probs[:, 0]
    p2 = true_class_probs[:, 1]
    
    result = ut.apply_distance_metric(p1, p2, metric)
    return result


#calcola la similarità tra distribuzioni di probabilità predittive in termini di correlazione ed errore 
#se concordano/discordano o meno, quanto sono simili le rispettive distribuzioni di probabilità?   
def probabilities_similarity(models, X_ts, y_true, metric = spearmanr, preds_concordance = None, **kwargs):

    _, probs = ut.predictions(models, X_ts) 
    probs1, probs2 = probs
    
    if preds_concordance is not None:
        #filtriamo per predizioni concordanti o discordanti
        m1_probs, m2_probs = ut.prediction_concordance_filter(models, X_ts, (probs1, probs2), concordant = preds_concordance)
    else:
        m1_probs, m2_probs = probs1, probs2
        
    results = []
    for class_ in range(m1_probs.shape[1]):
        result = ut.apply_distance_metric(m1_probs[:, class_], m2_probs[:, class_], metric)
        
        results.append(result)
    return np.array(results) #similarity per classe