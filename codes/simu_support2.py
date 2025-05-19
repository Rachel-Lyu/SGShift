from SGShift import *
import pandas as pd
import numpy as np
import os

reg_map = {
    'decision_tree': 'decision_tree_reg', 
    'svm':'svm_reg', 
    'logistic_regression':'linear_regression', 
    'mlp_classifier':'mlp_regressor', 
    'random_forest':'random_forest_reg', 
    'gb_classifier':'gb_regressor'
}

df = pd.read_csv('/net/dali/home/mscbio/rul98/PheOpt/compare/whyshift/datasets/support2_processed.csv')
source_df = df[df['hospdead']==0].drop(columns = ['hospdead'])
target_df = df[df['hospdead']==1].drop(columns = ['hospdead'])

feature_names = source_df.drop(columns = ['charges']).columns
feature_names = np.array(feature_names)
X_S = source_df[feature_names].values
y_S = np.log10(source_df['charges'].to_numpy())
target_X = target_df[feature_names].values
target_y = np.log10(target_df['charges'].to_numpy())

args = parse_arguments()
X_T, X_val, y_T, y_val = train_test_split(target_X, target_y, train_size = args.n_test, test_size = 800, random_state=42)
directory = args.directory + str(args.n_test)
if not os.path.exists(directory):
    os.makedirs(directory)
lambdas = np.geomspace(0.00001, 1, num = 21)
sel_thrs = np.linspace(0.05, 1.0, 20)
results_naive = []
results_absorption = []
ll1 = np.array([0.0003, 0.001, 0.003])
results_naive_knock = {l:[] for l in ll1}
ll2 = np.array([0.00003, 0.0001, 0.0003])
results_knock_absorption = {l:[] for l in ll2}
ll3 = np.geomspace(0.0000001, 1, num = 29)
results_diff = []
results_shap = []
results_whyshift = []
all_sel_whyshift = []
results_source = []
results_target = []
rng = check_random_state(args.random_state)
n_S, p = X_S.shape
n_T = X_T.shape[0]
n_val = X_val.shape[0]
y_T_val_freq = (y_T.sum() + y_val.sum()) / (len(y_T) + len(y_val))
shifted_feature = np.array(['age', 'slos', 'd.time', 'avtisst', 'hday', 
    'wblc', 'temp', 'dzgroup_CHF', 'dzclass_ARF/MOSF', 'dzclass_Cancer', 
    'income_>$50k', 'income_under $11k', 'race_white', 'sfdm2_Coma or Intub', 'ca_metastatic'])
shift_subset_y = np.where(np.isin(feature_names, shifted_feature))[0]
# if args.model == args.solver:
#     args.shift_magnitude_y = args.shift_magnitude_y / 5
for b in range(args.B):
    simulated = simulate_data(
        task=args.task,
        model_type=reg_map[args.model],
        X_S = X_S, 
        X_T_val = np.concatenate([X_T, X_val]),
        y_S_tmp = y_S, 
        y_T_val_freq = y_T_val_freq,
        n_S=n_S,
        n_T=n_T,
        n_val=n_val,
        p=p,
        p_informative=None,
        shift_features_X=None,
        shift_magnitude_X=0.,
        shift_subset_y=shift_subset_y,
        shift_magnitude_y=args.shift_magnitude_y,
        linear_shift_y = True,
        random_state=args.random_state+b
    )
    X_S = simulated['source']['X']
    y_S = simulated['source']['y']
    X_T = simulated['target']['X'] 
    y_T = simulated['target']['y']
    X_val = simulated['validation']['X']
    y_val = simulated['validation']['y']
    
    solver = fit_model(args.task, reg_map[args.solver], args.random_state)
    source_only = deepcopy(solver)
    source_only.fit(X_S, y_S)
    target_only = deepcopy(solver)
    target_only.fit(X_T, y_T)
    
    f_x_T = source_only.predict(X_T)
    f_x_val = source_only.predict(X_val)
    eval_results_source = evaluation(y_val, f_x_val, task = args.task)
    results_source.append(eval_results_source)
    eval_results_target = evaluation(y_val, target_only.predict(X_val), task = args.task)
    results_target.append(eval_results_target)
    pd.DataFrame(results_source).to_csv(f'{directory}/source_{args.model}_{args.solver}.csv', index=False)
    pd.DataFrame(results_target).to_csv(f'{directory}/target_{args.model}_{args.solver}.csv', index=False)
    
    for lambda_reg in lambdas:
        delta_estimated = estimate_delta(X_T, y_T, lambda_reg, source_only, task = args.task)
        selected_naive = np.where(np.abs(delta_estimated) > 1e-4)[0]
        x_delta = X_val @ delta_estimated
        eval_results_naive = evaluation(y_val, f_x_val + x_delta, shift_subset_y, selected_naive, task = args.task)
        eval_results_naive['lambda_reg'] = lambda_reg
        results_naive.append(eval_results_naive)
        df = pd.DataFrame(results_naive)
        df.to_csv(f'{directory}/naive_{args.model}_{args.solver}.csv', index=False)

    pi = derandom_knock(solver, X_S, y_S, X_T, y_T, 20, ll1, task = args.task)
    for l_ in ll1:
        for sel_thr in sel_thrs:
            selected_naive_knock = np.where(pi[l_] >= sel_thr)[0]
            if len(selected_naive_knock) > 0: 
                delta_estimated = estimate_delta(X_T[:, selected_naive_knock], y_T, l_, f_x = f_x_T, task = args.task)
                x_delta = X_val[:, selected_naive_knock] @ np.array(delta_estimated)
                shifted_log_odds = f_x_val + x_delta
            else: 
                shifted_log_odds = f_x_val
            eval_results_naive_knock = evaluation(y_val, shifted_log_odds, shift_subset_y, selected_naive_knock, task = args.task)
            eval_results_naive_knock['sel_thr'] = sel_thr
            results_naive_knock[l_].append(eval_results_naive_knock)
            df_results_naive_knock = pd.DataFrame(results_naive_knock[l_])
            df_results_naive_knock.to_csv(f'{directory}/naive_knock_{args.model}_{args.solver}_{l_}.csv', index=False)
    
    for lambda_reg in lambdas:
        lambda_reg = lambda_reg / 10
        lambda_beta = 0.9 * lambda_reg
        beta_est, delta_est = estimate_beta_delta(X_S, y_S, X_T, y_T, lambda_beta, lambda_reg, source_only, task = args.task)
        selected_miss = np.where(np.abs(delta_est) > 1e-4)[0]
        x_beta = X_val @ beta_est
        x_delta = X_val @ delta_est
        shifted_log_odds = f_x_val + x_beta + x_delta
        eval_results_absorption = evaluation(y_val, shifted_log_odds, shift_subset_y, selected_miss, task = args.task)
        eval_results_absorption['lambda_reg'] = lambda_reg
        results_absorption.append(eval_results_absorption)
        df = pd.DataFrame(results_absorption)
        df.to_csv(f'{directory}/absorption_{args.model}_{args.solver}.csv', index=False)
    
    pi = derandom_knock_misspecified(solver, X_S, y_S, X_T, y_T, 20, ll2, task = args.task)
    for l_ in ll2:
        beta_estimated, delta_estimated = estimate_beta_delta(X_S, y_S, X_T, y_T, 0.9*l_, l_, source_only, task = args.task)
        for sel_thr in sel_thrs:
            selected_knock_absorption = np.where(pi[l_] >= sel_thr)[0]
            if len(selected_knock_absorption) > 0: 
                delta_estimated = estimate_delta(X_T[:, selected_knock_absorption], y_T, l_, f_x = (f_x_T + X_T @ np.array(beta_estimated)), task = args.task)
                x_delta = X_val[:, selected_knock_absorption] @ np.array(delta_estimated)
                shifted_log_odds = f_x_val + X_val @ np.array(beta_estimated) + x_delta
            else:
                shifted_log_odds = f_x_val
            eval_results_knock_absorption = evaluation(y_val, shifted_log_odds, shift_subset_y, selected_knock_absorption, task = args.task)
            eval_results_knock_absorption['sel_thr'] = sel_thr
            results_knock_absorption[l_].append(eval_results_knock_absorption)
            df_results_knock_absorption = pd.DataFrame(results_knock_absorption[l_])
            df_results_knock_absorption.to_csv(f'{directory}/absorption_knock_{args.model}_{args.solver}_{l_}.csv', index=False)
    
    X_S_train, X_S_test, y_S_train, y_S_test = train_test_split(X_S, y_S, test_size=0.5, random_state=args.random_state)
    X_T_train, X_T_test, y_T_train, y_T_test = train_test_split(X_T, y_T, test_size=0.5, random_state=args.random_state)
    source_only.fit(X_S_train, y_S_train)
    target_only.fit(X_T_train, y_T_train)
    
    X_stack = np.vstack([X_S_test, X_T_test])
    f_diff = target_only.predict(X_stack) - source_only.predict(X_stack)
    
    for l_ in ll3:
        diff_model = Lasso(alpha=l_, max_iter=10000, random_state=args.random_state).fit(X_stack, f_diff)
        diff_estimated = diff_model.coef_
        selected_diff = np.where(abs(diff_estimated) > 1e-4)[0]
        shifted_log_odds = f_x_val + X_val @ np.array(diff_estimated)
        eval_results_diff = evaluation(y_val, shifted_log_odds, shift_subset_y, selected_diff, task = args.task)
        eval_results_diff['lambda_reg'] = l_
        results_diff.append(eval_results_diff)
        df_results_diff = pd.DataFrame(results_diff)
        df_results_diff.to_csv(f'{directory}/diff_{args.model}_{args.solver}.csv', index=False)
        
    explainer_source = shap.Explainer(source_only.predict, X_stack)
    phi_S = explainer_source(X_stack).values
    explainer_target = shap.Explainer(target_only.predict, X_stack)
    phi_T = explainer_target(X_stack).values
    delta_phi = phi_S - phi_T    # shape (n_samples, n_features)
    abs_gap   = np.abs(delta_phi).mean(0)
    sort_gap = np.sort(abs_gap)[::-1]
    for i_thr, sel_thr in enumerate(sort_gap):
        selected_shap = np.where(abs_gap >= sel_thr)[0]
        true = set(shift_subset_y)
        selected = set(selected_shap)
        tp = len(selected & true)  # True Positives
        fp = len(selected - true)  # False Positives
        fn = len(true - selected)  # False Negatives
        eval_results_shap = {'TP': tp, 'FP': fp, 'FN': fn, 'sel_thr': i_thr / len(sort_gap)}
        results_shap.append(eval_results_shap)
    df_results_shap = pd.DataFrame(results_shap)
    df_results_shap.to_csv(f'{directory}/shap_{args.model}_{args.solver}.csv', index=False)
        
    # wA, wB, new_X, new_weights = shared_reweight(X_S_test, X_T_test)
    proba_P2Q = source_only.predict(X_stack)
    proba_Q2P = target_only.predict(X_stack)
    new_Y = np.abs(proba_P2Q - proba_Q2P)

    region_model = DecisionTreeRegressor(max_depth=6,min_samples_leaf=100,min_samples_split=200,min_weight_fraction_leaf=0.05, ccp_alpha=0.0001, random_state=args.random_state+b).fit(X_stack, new_Y)# , sample_weight=new_weights)
    S = np.where(region_model.feature_importances_>0)[0]
    sel_whyshift = np.zeros(X_S.shape[1])
    sel_whyshift[S] = 1
    all_sel_whyshift.append(sel_whyshift)
    if b % 5 == 4:
        all_sel_whyshift = all_sel_whyshift[-5:]
        selfreq_whyshift = np.array(all_sel_whyshift).mean(axis = 0)
        for sel_thr in np.linspace(0.2, 1.0, 5):
            selected_whyshift = np.where(selfreq_whyshift >= sel_thr)[0]
            true = set(shift_subset_y)
            selected = set(selected_whyshift)
            tp = len(selected & true)  # True Positives
            fp = len(selected - true)  # False Positives
            fn = len(true - selected)  # False Negatives
            eval_results_whyshift = {'TP': tp, 'FP': fp, 'FN': fn, 'sel_thr': sel_thr}
            results_whyshift.append(eval_results_whyshift)
        df_results_whyshift = pd.DataFrame(results_whyshift)
        df_results_whyshift.to_csv(f'{directory}/whyshift_{args.model}_{args.solver}.csv', index=False)